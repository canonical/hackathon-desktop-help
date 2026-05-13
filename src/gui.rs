use anyhow::Result;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;
use libadwaita::prelude::*;

use crate::llm::{CopilotClient, LlmClient, Message, OllamaClient};
use crate::{DEFAULT_OLLAMA_URL, SYSTEM_PROMPT};

const APP_ID: &str = "com.canonical.UbuntuDesktopHelp";

// Messages sent from the LLM worker thread to the GTK main loop.
enum StreamMsg {
    Token(String),
    Error(String),
    Done,
}

// Entry point for `ubuntu-desktop-help gui <query>`. Starts the GTK main loop
// and presents a single window that streams the answer for `query`.
pub fn run(query: String, use_copilot: bool, model: String) -> Result<()> {
    let app = adw::Application::builder().application_id(APP_ID).build();

    // connect_activate is called when the application is told to present itself.
    // We pass an empty argv to .run_with_args because clap has already consumed
    // the real argv and GTK would otherwise re-parse it.
    app.connect_activate(move |app| {
        build_window(app, query.clone(), use_copilot, model.clone());
    });

    let empty: [&str; 0] = [];
    app.run_with_args(&empty);
    Ok(())
}

fn build_window(app: &adw::Application, query: String, use_copilot: bool, model: String) {
    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Ubuntu Desktop Help")
        .default_width(640)
        .default_height(520)
        .build();

    // ToolbarView gives us a standard GNOME header bar above the content area.
    let toolbar = adw::ToolbarView::new();
    let header = adw::HeaderBar::new();
    toolbar.add_top_bar(&header);

    // Vertical box: the user's question on top, the streamed answer below.
    let vbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(8)
        .margin_top(16)
        .margin_bottom(16)
        .margin_start(16)
        .margin_end(16)
        .build();

    let header_text = if query.is_empty() {
        "Type your question in the GNOME overview after <tt>??</tt>".to_string()
    } else {
        format!("<b>{}</b>", glib::markup_escape_text(&query))
    };
    let question_label = gtk4::Label::builder()
        .label(&header_text)
        .use_markup(true)
        .xalign(0.0)
        .wrap(true)
        .wrap_mode(gtk4::pango::WrapMode::WordChar)
        .build();
    vbox.append(&question_label);

    // Spinner shown until the first token arrives. Hidden when there's nothing to ask.
    let spinner = gtk4::Spinner::builder().spinning(true).build();
    let spinner_row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .build();
    spinner_row.append(&spinner);
    let thinking = gtk4::Label::new(Some("Thinking…"));
    spinner_row.append(&thinking);
    if query.is_empty() {
        spinner_row.set_visible(false);
    }
    vbox.append(&spinner_row);

    // TextView holds the streamed answer. It starts empty and is filled
    // token-by-token as the model responds.
    let text_view = gtk4::TextView::builder()
        .editable(false)
        .cursor_visible(false)
        .wrap_mode(gtk4::WrapMode::WordChar)
        .top_margin(8)
        .bottom_margin(8)
        .left_margin(4)
        .right_margin(4)
        .build();
    let buffer = text_view.buffer();

    let scroll = gtk4::ScrolledWindow::builder()
        .vexpand(true)
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .child(&text_view)
        .build();
    vbox.append(&scroll);

    toolbar.set_content(Some(&vbox));
    window.set_content(Some(&toolbar));
    window.present();

    // Channel from the LLM worker (tokio thread) to the GTK main loop. Only
    // start a worker if there's actually something to ask; an empty-query
    // launch (from the app grid) just shows the placeholder above.
    let (tx, rx) = async_channel::unbounded::<StreamMsg>();
    if !query.is_empty() {
        spawn_worker(query, use_copilot, model, tx);
    } else {
        let _ = tx.send_blocking(StreamMsg::Done);
    }

    // Drain the channel on the GTK main loop and update the buffer in place.
    glib::spawn_future_local(async move {
        let mut got_first_token = false;
        while let Ok(msg) = rx.recv().await {
            match msg {
                StreamMsg::Token(t) => {
                    if !got_first_token {
                        spinner_row.set_visible(false);
                        got_first_token = true;
                    }
                    let mut end = buffer.end_iter();
                    buffer.insert(&mut end, &t);
                }
                StreamMsg::Error(e) => {
                    spinner_row.set_visible(false);
                    let mut end = buffer.end_iter();
                    buffer.insert(&mut end, &format!("Error: {e}"));
                }
                StreamMsg::Done => break,
            }
        }
    });
}

// Spawns a thread that runs a single-threaded tokio runtime, executes one
// chat call, and forwards tokens to the GTK side via `tx`. We use a fresh
// thread because GTK and tokio each want to own the main thread.
fn spawn_worker(
    query: String,
    use_copilot: bool,
    model: String,
    tx: async_channel::Sender<StreamMsg>,
) {
    std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                let _ = tx.send_blocking(StreamMsg::Error(format!("runtime: {e}")));
                let _ = tx.send_blocking(StreamMsg::Done);
                return;
            }
        };

        rt.block_on(async move {
            let client = if use_copilot {
                match CopilotClient::create().await {
                    Ok(c) => LlmClient::Copilot(c),
                    Err(e) => {
                        let _ = tx.send(StreamMsg::Error(format!("auth: {e}"))).await;
                        let _ = tx.send(StreamMsg::Done).await;
                        return;
                    }
                }
            } else {
                LlmClient::Ollama(OllamaClient::new(DEFAULT_OLLAMA_URL.to_string(), model))
            };

            let messages = vec![
                Message {
                    role: "system".into(),
                    content: SYSTEM_PROMPT.into(),
                },
                Message {
                    role: "user".into(),
                    content: query,
                },
            ];

            let tx_token = tx.clone();
            let on_first_token = || {};
            let on_token = |t: &str| {
                // send_blocking on an unbounded channel only blocks if the
                // receiver was dropped, in which case we just give up.
                let _ = tx_token.send_blocking(StreamMsg::Token(t.to_string()));
            };
            let result = client.chat(&messages, on_first_token, on_token).await;
            if let Err(e) = result {
                let _ = tx.send(StreamMsg::Error(e.to_string())).await;
            }
            let _ = tx.send(StreamMsg::Done).await;
        });
    });
}
