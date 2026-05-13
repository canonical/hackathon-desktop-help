use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use text_splitter::TextSplitter;

// Maximum characters per chunk; keeps each chunk within a useful slice of LLM context
const CHUNK_SIZE: usize = 512;
// Output dimension of BGE-small-en-v1.5; written into the index header so the runtime can verify
const EMBEDDING_DIM: usize = 384;

// Documentation repositories cloned into docs/ at build time.
// Each tuple is (clone URL, subdirectory name under docs/).
const DOCS_REPOS: &[(&str, &str)] = &[
    ("https://github.com/ubuntu/ubuntu-desktop-documentation", "ubuntu-desktop-documentation"),
    // ("https://github.com/canonical/ubuntu-server-documentation", "ubuntu-server-documentation"),
    // ("https://github.com/canonical/ubuntu-core-docs", "ubuntu-core-docs"),
];

fn main() -> anyhow::Result<()> {
    // Only re-run this build script when build.rs itself changes.
    // We intentionally do NOT watch docs/ here: since we clone into docs/ ourselves,
    // watching it would cause an infinite rebuild loop.
    println!("cargo:rerun-if-changed=build.rs");

    clone_or_update_repos("docs")?;

    let out_dir = env::var("OUT_DIR")?;
    let index_path = Path::new(&out_dir).join("index.bin");

    let chunks = load_chunks("docs");

    if chunks.is_empty() {
        println!("cargo:warning=No markdown files found in docs/; RAG index will be empty.");
        write_index(&index_path, EMBEDDING_DIM, &[], &[])?;
        return Ok(());
    }

    println!(
        "cargo:warning=Building RAG index from {} chunks (BGE-small model downloads ~130 MB on first run)…",
        chunks.len()
    );

    let mut embedder = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
    )?;

    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = embedder.embed(texts, None)?;

    write_index(&index_path, EMBEDDING_DIM, &chunks, &embeddings)?;

    println!(
        "cargo:warning=RAG index ready: {} vectors ({} dims).",
        chunks.len(),
        EMBEDDING_DIM
    );

    Ok(())
}

// Ensures every repo in DOCS_REPOS is present under `docs_dir`.
// Clones with --depth 1 on first run; does `git pull --ff-only` on subsequent runs.
fn clone_or_update_repos(docs_dir: &str) -> anyhow::Result<()> {
    fs::create_dir_all(docs_dir)?;
    for (url, name) in DOCS_REPOS {
        let dest = Path::new(docs_dir).join(name);
        if dest.join(".git").is_dir() {
            println!("cargo:warning=Updating {name}…");
            let status = Command::new("git")
                .args(["-C", dest.to_str().unwrap(), "pull", "--ff-only", "--quiet"])
                .status()?;
            if !status.success() {
                println!("cargo:warning=Warning: `git pull` failed for {name}; using existing checkout.");
            }
        } else {
            println!("cargo:warning=Cloning {url} into docs/{name}…");
            let status = Command::new("git")
                .args(["clone", "--depth", "1", "--quiet", url, dest.to_str().unwrap()])
                .status()?;
            if !status.success() {
                anyhow::bail!("Failed to clone {url}");
            }
            println!("cargo:warning=Cloned {name}.");
        }
    }
    Ok(())
}

struct Chunk {
    source: String,
    text: String,
}

// Walks `dir` recursively for .md files, strips markdown to plain text, and splits into chunks.
fn load_chunks(dir: &str) -> Vec<Chunk> {
    let mut md_files = Vec::new();
    collect_md_files(Path::new(dir), &mut md_files);
    // Sort for a deterministic index regardless of filesystem ordering
    md_files.sort();

    let splitter = TextSplitter::new(CHUNK_SIZE);
    let mut chunks = Vec::new();

    for file_path in &md_files {
        let raw = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let plain = markdown_to_plain_text(&raw);
        let source = file_path.display().to_string();
        for chunk_text in splitter.chunks(&plain) {
            let text = chunk_text.trim().to_string();
            if !text.is_empty() {
                chunks.push(Chunk { source: source.clone(), text });
            }
        }
    }

    chunks
}

// Recursively collects all .md file paths under `dir` into `out`.
fn collect_md_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(iter) => iter,
        Err(_) => return,
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_md_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("md") {
            out.push(path);
        }
    }
}

// Strips markdown syntax to plain text using pulldown-cmark's event stream.
fn markdown_to_plain_text(markdown: &str) -> String {
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(markdown, opts);
    let mut text = String::new();
    for event in parser {
        match event {
            Event::Text(t) | Event::Code(t) => text.push_str(&t),
            Event::SoftBreak | Event::HardBreak => text.push(' '),
            Event::End(TagEnd::Paragraph)
            | Event::End(TagEnd::Heading { .. })
            | Event::End(TagEnd::Item)
            | Event::End(TagEnd::CodeBlock) => text.push('\n'),
            Event::Start(Tag::CodeBlock(_)) => text.push('\n'),
            _ => {}
        }
    }
    text
}

// Binary index format written to $OUT_DIR/index.bin and embedded by include_bytes! at runtime:
//   dim       u64 le   — embedding dimension (384 for BGE-small)
//   n_chunks  u64 le   — number of entries
//   per entry:
//     src_len u64 le + src_bytes   — source file path
//     txt_len u64 le + txt_bytes   — chunk plain text
//     dim × f32 le                 — embedding vector
fn write_index(
    path: &Path,
    dim: usize,
    chunks: &[Chunk],
    embeddings: &[Vec<f32>],
) -> anyhow::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&(dim as u64).to_le_bytes())?;
    f.write_all(&(chunks.len() as u64).to_le_bytes())?;
    for (chunk, vec) in chunks.iter().zip(embeddings.iter()) {
        let src = chunk.source.as_bytes();
        f.write_all(&(src.len() as u64).to_le_bytes())?;
        f.write_all(src)?;
        let txt = chunk.text.as_bytes();
        f.write_all(&(txt.len() as u64).to_le_bytes())?;
        f.write_all(txt)?;
        for val in vec {
            f.write_all(&val.to_le_bytes())?;
        }
    }
    Ok(())
}
