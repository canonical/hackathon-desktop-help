use std::collections::HashMap;

use anyhow::{Context, Result};
use zbus::{connection, interface};
use zbus::zvariant::{OwnedValue, Value};

// The unique D-Bus bus name and object path GNOME Shell looks up via the
// search-provider .ini file. Both must match the .ini exactly.
const BUS_NAME: &str = "com.canonical.UbuntuDesktopHelp";
const OBJECT_PATH: &str = "/com/canonical/UbuntuDesktopHelp/SearchProvider";

// Only intercept overview searches that begin with this prefix, so we don't
// match every keystroke the user types into the overview.
const TRIGGER_PREFIX: &str = "??";

// Single fixed result identifier — the search provider always returns either
// zero results or this one "open the assistant" entry.
const RESULT_ID: &str = "ask";

struct UbuntuDesktopHelp;

impl UbuntuDesktopHelp {
    // Joins the search terms and, if the query starts with the trigger prefix,
    // returns the question with the prefix stripped. Otherwise returns None.
    fn extract_query(terms: &[String]) -> Option<String> {
        let joined = terms.join(" ");
        let trimmed = joined.trim_start();
        let rest = trimmed.strip_prefix(TRIGGER_PREFIX)?;
        let q = rest.trim();
        if q.is_empty() { None } else { Some(q.to_string()) }
    }
}

#[interface(name = "org.gnome.Shell.SearchProvider2")]
impl UbuntuDesktopHelp {
    // GNOME calls this with the user's terms split on whitespace. We return
    // a single placeholder result if the trigger prefix is present.
    async fn get_initial_result_set(&self, terms: Vec<String>) -> Vec<String> {
        if Self::extract_query(&terms).is_some() {
            vec![RESULT_ID.to_string()]
        } else {
            Vec::new()
        }
    }

    // Called as the user continues typing. Same trigger check applies.
    async fn get_subsearch_result_set(
        &self,
        _previous_results: Vec<String>,
        terms: Vec<String>,
    ) -> Vec<String> {
        if Self::extract_query(&terms).is_some() {
            vec![RESULT_ID.to_string()]
        } else {
            Vec::new()
        }
    }

    // Metadata GNOME uses to render each result tile. We only ever return
    // metadata for our single fixed identifier; the description shows the
    // current question so the user sees what they're about to send.
    async fn get_result_metas(
        &self,
        identifiers: Vec<String>,
    ) -> Vec<HashMap<String, OwnedValue>> {
        identifiers
            .into_iter()
            .filter(|id| id == RESULT_ID)
            .map(|id| {
                let mut meta = HashMap::new();
                meta.insert("id".to_string(), str_value(&id));
                meta.insert("name".to_string(), str_value("Ask Ubuntu Desktop Help"));
                meta.insert(
                    "description".to_string(),
                    str_value("Open the assistant for an answer"),
                );
                meta.insert("gicon".to_string(), str_value("help-browser"));
                meta
            })
            .collect()
    }

    // GNOME calls this when the user clicks our result. We launch a separate
    // GUI process so the search provider stays small and unblocked.
    async fn activate_result(
        &self,
        _identifier: String,
        terms: Vec<String>,
        _timestamp: u32,
    ) {
        if let Some(query) = Self::extract_query(&terms) {
            spawn_gui(&query);
        }
    }

    // GNOME calls this if the user hits enter without selecting a tile.
    async fn launch_search(&self, terms: Vec<String>, _timestamp: u32) {
        if let Some(query) = Self::extract_query(&terms) {
            spawn_gui(&query);
        }
    }
}

// Builds an OwnedValue containing a string, with the unwrap path that can't
// fail in practice — string conversion is infallible inside zvariant.
fn str_value(s: &str) -> OwnedValue {
    Value::from(s).try_to_owned().expect("string OwnedValue")
}

// Launches `ubuntu-desktop-help gui <query>` in the background. We use the
// path of the currently running binary so the dev build, the installed binary,
// and the snap all do the right thing without configuration.
fn spawn_gui(query: &str) {
    let exe = std::env::current_exe().unwrap_or_else(|_| "ubuntu-desktop-help".into());
    if let Err(e) = std::process::Command::new(&exe)
        .arg("gui")
        .arg(query)
        .spawn()
    {
        eprintln!("failed to spawn gui process: {e}");
    }
}

// Connects to the session bus, registers our object, and blocks forever.
// D-Bus activation will start this process when GNOME first opens the overview
// after login; the connection is kept alive by holding the Connection handle.
pub async fn run() -> Result<()> {
    let _conn = connection::Builder::session()
        .context("failed to open session D-Bus")?
        .name(BUS_NAME)?
        .serve_at(OBJECT_PATH, UbuntuDesktopHelp)?
        .build()
        .await
        .context("failed to register search provider on the bus")?;

    // Hold the connection open for the lifetime of the process.
    std::future::pending::<()>().await;
    Ok(())
}
