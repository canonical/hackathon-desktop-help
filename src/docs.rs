use std::fs;
use std::path::Path;

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use text_splitter::MarkdownSplitter;

// Maximum number of characters per chunk; keeps each chunk within a typical LLM context window slice
const CHUNK_SIZE: usize = 512;

// A single piece of documentation text with its source file path
pub struct Chunk {
    // The file the chunk came from, used to label context in the system prompt
    pub source: String,
    // Plain-text content of this chunk (markdown syntax stripped)
    pub text: String,
}

// Walks `dir` for .md files, strips markdown syntax, splits into chunks, and returns them all.
// Files or directories that cannot be read are skipped with a warning on stderr.
pub fn load_chunks(dir: &str) -> Vec<Chunk> {
    let path = Path::new(dir);

    if !path.is_dir() {
        eprintln!("Warning: docs directory '{dir}' not found; proceeding without documentation context.");
        return Vec::new();
    }

    let mut entries: Vec<_> = match fs::read_dir(path) {
        Ok(iter) => iter.filter_map(|e| e.ok()).collect(),
        Err(e) => {
            eprintln!("Warning: could not read docs directory '{dir}': {e}");
            return Vec::new();
        }
    };

    // Sort for deterministic ordering across runs
    entries.sort_by_key(|e| e.file_name());

    let splitter = MarkdownSplitter::new(CHUNK_SIZE);
    let mut chunks = Vec::new();

    for entry in entries {
        let entry_path = entry.path();

        // Only process markdown files
        if entry_path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }

        let raw = match fs::read_to_string(&entry_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: could not read '{}': {e}", entry_path.display());
                continue;
            }
        };

        // Strip markdown syntax to plain text so the LLM receives clean prose
        let plain = markdown_to_plain_text(&raw);
        let source = entry_path.display().to_string();

        // Split the plain text into bounded chunks; collect references into owned Strings
        for chunk_text in splitter.chunks(&plain) {
            let text = chunk_text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            chunks.push(Chunk {
                source: source.clone(),
                text,
            });
        }
    }

    chunks
}

// Converts markdown to plain text by walking the pulldown-cmark event stream and
// keeping only text content, discarding all markup tags and metadata.
fn markdown_to_plain_text(markdown: &str) -> String {
    // TABLES and STRIKETHROUGH are common in Ubuntu docs
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(markdown, opts);
    let mut text = String::new();

    for event in parser {
        match event {
            // Inline text and inline code both contribute readable content
            Event::Text(t) | Event::Code(t) => text.push_str(&t),
            // Soft and hard breaks become spaces to avoid run-on words
            Event::SoftBreak | Event::HardBreak => text.push(' '),
            // End of a block element: add a newline to preserve paragraph separation
            Event::End(TagEnd::Paragraph)
            | Event::End(TagEnd::Heading { .. })
            | Event::End(TagEnd::Item)
            | Event::End(TagEnd::CodeBlock) => text.push('\n'),
            // Code block content is captured by Event::Text above; just add spacing after
            Event::Start(Tag::CodeBlock(_)) => text.push('\n'),
            _ => {}
        }
    }

    text
}
