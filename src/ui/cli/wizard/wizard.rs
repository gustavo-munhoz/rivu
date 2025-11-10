use anyhow::{Context, Result};
use inquire::Select;
use serde_json::{Map, Value};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use strum::{EnumMessage, IntoEnumIterator};

use crate::ui::cli::drivers::PromptDriver;
use crate::ui::types::choices::{FieldKind, UIChoice, schema_for, specs_for_kind};

const DIM_ITALIC: &str = "\x1b[2m\x1b[3m";
const RESET: &str = "\x1b[0m";

struct KindItem<K> {
    kind: K,
    text: String,
}

impl<K> Display for KindItem<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text)
    }
}

fn kind_items<K>() -> Vec<KindItem<K>>
where
    K: Copy + Into<&'static str> + EnumMessage + IntoEnumIterator,
{
    K::iter()
        .map(|k| {
            let label = k.get_message().unwrap_or_else(|| k.into());
            let desc = k.get_detailed_message().unwrap_or("");
            let text = if desc.is_empty() {
                label.to_string()
            } else {
                format!("{label}  {DIM_ITALIC}{desc}{RESET}")
            };
            KindItem { kind: k, text }
        })
        .collect()
}

pub fn prompt_choice<C: UIChoice, D: PromptDriver>(driver: &D) -> Result<C> {
    // 1) Choose variant/kind
    let items = kind_items::<C::Kind>();
    let mut select = inquire::Select::new(C::prompt_label(), items);
    if let Some(help) = C::prompt_help() {
        select = select.with_help_message(help);
    }
    let selected = select.prompt()?;
    let choice_kind: C::Kind = selected.kind;

    // 2) Load schema + field specs for chosen branch
    let key: &'static str = choice_kind.into();
    let schema = schema_for::<C>();
    let specs = specs_for_kind(&schema, key)?;
    let defaults = C::default_params(choice_kind);

    // 3) Prompt each field
    let mut params = Map::new();
    for s in specs {
        let init = s.default.clone().or_else(|| defaults.get(&s.name).cloned());
        let help = s.description.as_deref().unwrap_or("");

        // numeric Option<T> with "leave blank for none"
        let is_optional_numeric = !s.required
            && matches!(s.kind, FieldKind::Integer | FieldKind::Number)
            && matches!(init, None | Some(Value::Null));

        let val_opt: Option<Value> = if is_optional_numeric {
            // show prefilled text, blank -> None
            let def_txt = match s.kind {
                FieldKind::Integer => init
                    .as_ref()
                    .and_then(|v| v.as_u64())
                    .map(|n| n.to_string()),
                FieldKind::Number => init
                    .as_ref()
                    .and_then(|v| v.as_f64())
                    .map(|x| x.to_string()),
                _ => None,
            }
                .unwrap_or_default();

            let answer = driver.ask_string(
                &s.title,
                &format!("{help}\n(leave blank for none)"),
                &def_txt,
            )?;

            let answer = answer.trim();
            if answer.is_empty() {
                None
            } else {
                Some(match s.kind {
                    FieldKind::Integer => {
                        let n: u64 = answer
                            .parse()
                            .with_context(|| format!("invalid integer for {}", s.title))?;
                        Value::from(n)
                    }
                    FieldKind::Number => {
                        let x: f64 = answer
                            .parse()
                            .with_context(|| format!("invalid number for {}", s.title))?;
                        Value::from(x)
                    }
                    _ => unreachable!(),
                })
            }
        } else {
            // all other cases
            Some(match s.kind {
                FieldKind::Boolean => {
                    let def = init.and_then(|v| v.as_bool()).unwrap_or(false);
                    Value::Bool(driver.ask_bool(&s.title, help, def)?)
                }

                FieldKind::String => {
                    // If schema has an enum, show a Select
                    if let Some(opts) = &s.allowed {
                        // build the menu (clone allowed values)
                        let mut menu = opts.clone();

                        // add a "none" entry for optional string-enums
                        let mut none_idx: Option<usize> = None;
                        if !s.required {
                            none_idx = Some(menu.len());
                            menu.push("— none —".to_string());
                        }

                        // compute starting index from default/init
                        let def_str = init.as_ref().and_then(|v| v.as_str());
                        let mut start_idx = def_str
                            .and_then(|cur| menu.iter().position(|o| o == cur))
                            .unwrap_or(0);

                        // if default is None/null and we added "none", start there
                        if def_str.is_none() {
                            if let Some(idx) = none_idx {
                                start_idx = idx;
                            }
                        }

                        let selected = Select::new(&s.title, menu.clone())
                            .with_help_message(help)
                            .with_starting_cursor(start_idx.min(menu.len().saturating_sub(1)))
                            .prompt()?;

                        if let Some(idx) = none_idx {
                            if selected == "— none —" && start_idx == idx {
                                // treat "none" as absence; skip insert by returning None
                                // (the outer `if let Some(val)` will just not insert)
                                // If you prefer explicit null, return Some(Value::Null) here.
                                None
                            } else if selected == "— none —" {
                                None
                            } else {
                                Some(Value::String(selected))
                            }
                        } else {
                            Some(Value::String(selected))
                        }
                            .unwrap_or_else(|| Value::Null) // keep a consistent type (optional)
                    } else {
                        // Free-text string. Special-case ARFF path validation.
                        let def = init
                            .and_then(|v| v.as_str().map(|s| s.to_string()))
                            .unwrap_or_default();

                        let is_arff_path = s.name == "path";
                        let answered = if is_arff_path {
                            let more_help = if help.is_empty() {
                                "Please type a valid .arff file path"
                            } else {
                                help
                            };
                            let pb = prompt_path_until_ok(
                                driver,
                                &s.title,
                                more_help,
                                &def,
                                true,   // must_exist
                                true,   // must_be_file
                                &["arff"],
                            )?;
                            pb.to_string_lossy().into_owned()
                        } else {
                            driver.ask_string(&s.title, help, &def)?
                        };
                        Value::String(answered)
                    }
                }

                FieldKind::Integer => {
                    let def = init.and_then(|v| v.as_u64()).unwrap_or(0);
                    Value::from(driver.ask_u64(
                        &s.title,
                        help,
                        def,
                        s.min.map(|x| x as u64),
                        s.max.map(|x| x as u64),
                    )?)
                }

                FieldKind::Number => {
                    let def = init.and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Value::from(driver.ask_f64(&s.title, help, def, s.min, s.max)?)
                }
            })
        };

        if let Some(val) = val_opt {
            // Skip inserting explicit nulls if you want "unset" instead
            // if val.is_null() { /* skip */ } else { params.insert(...) }
            params.insert(s.name.clone(), val);
        }
    }

    // 4) Nested subprompts (e.g., learner/stream/evaluator)
    if let Some(extra) = C::subprompts(driver, choice_kind)? {
        params.extend(extra);
    }

    // 5) Build the final choice
    C::from_parts(choice_kind, Value::Object(params))
}

fn validate_path_str(
    input: &str,
    must_exist: bool,
    must_be_file: bool,
    allowed_exts: &[&str],
) -> Result<(), String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Path cannot be empty".into());
    }
    let p = Path::new(trimmed);

    if must_exist && !p.exists() {
        return Err(format!("Path does not exist: {}", p.display()));
    }
    if must_be_file && p.exists() && !p.is_file() {
        return Err("Expected a file path, not a directory".into());
    }
    if !allowed_exts.is_empty() {
        match p.extension().and_then(|e| e.to_str()) {
            Some(ext) if allowed_exts.iter().any(|e| e.eq_ignore_ascii_case(&ext)) => {}
            _ => return Err(format!("Expected a .{} file", allowed_exts.join(" / ."))),
        }
    }
    Ok(())
}

fn prompt_path_until_ok<D: PromptDriver>(
    driver: &D,
    title: &str,
    help: &str,
    default: &str,
    must_exist: bool,
    must_be_file: bool,
    allowed_exts: &[&str],
) -> Result<PathBuf> {
    loop {
        let answer = driver.ask_string(title, help, default)?;
        match validate_path_str(&answer, must_exist, must_be_file, allowed_exts) {
            Ok(()) => return Ok(PathBuf::from(answer)),
            Err(msg) => {
                eprintln!("✗ {}", msg);
            }
        }
    }
}
