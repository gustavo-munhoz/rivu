use anyhow::{Context, Result, anyhow, bail};
use schemars::{Schema, schema_for};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    String,
    Integer,
    Number,
    Boolean,
}

#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub title: String,
    pub description: Option<String>,
    pub required: bool,
    pub kind: FieldKind,
    pub default: Option<Value>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub allowed: Option<Vec<String>>,
}

// Return the whole tagged-enum schema for T
pub fn schema_for<T: schemars::JsonSchema>() -> Schema {
    schema_for!(T)
}

pub fn specs_for_kind(root: &Schema, kind_key: &str) -> Result<Vec<FieldSpec>> {
    let root_obj = root.as_object().context("root schema is not an object")?;

    let alts = root_obj
        .get("oneOf")
        .or_else(|| root_obj.get("anyOf"))
        .and_then(|v| v.as_array())
        .context("missing oneOf/anyOf")?;

    for branch in alts {
        let bobj = branch.as_object().context("branch is not object")?;
        let props = match bobj.get("properties").and_then(|v| v.as_object()) {
            Some(p) => p,
            None => continue,
        };

        if !discriminant_matches(props, kind_key) {
            continue;
        }

        let params_val = match props.get("params") {
            None => return Ok(vec![]),
            Some(v) => v,
        };

        let mut params_obj = match params_val.as_object() {
            Some(o) => o,
            None => return Ok(vec![]),
        };

        params_obj = match resolve_ref_obj(root_obj, params_obj) {
            Some(o) => o,
            None => return Ok(vec![]),
        };

        let Some(params_props) = params_obj.get("properties").and_then(|v| v.as_object()) else {
            return Ok(vec![]);
        };

        let required: Vec<String> = params_obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        let mut out = Vec::new();
        for (name, field_schema) in params_props {
            let mut fs_obj = field_schema
                .as_object()
                .context("field schema not object")?;

            if fs_obj.get("$ref").is_some() {
                fs_obj = resolve_ref_obj(root_obj, fs_obj)
                    .ok_or_else(|| anyhow!("failed to resolve field $ref for '{name}'"))?;
            }

            let title = fs_obj
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(name)
                .to_string();

            let description = fs_obj
                .get("description")
                .and_then(|v| v.as_str())
                .map(str::to_string);

            let default = fs_obj.get("default").cloned();

            let Some(kind) = detect_field_kind(fs_obj.get("type")) else {
                continue;
            };

            let min = fs_obj
                .get("minimum")
                .or_else(|| fs_obj.get("exclusiveMinimum"))
                .and_then(|v| v.as_f64());

            let max = fs_obj
                .get("maximum")
                .or_else(|| fs_obj.get("exclusiveMaximum"))
                .and_then(|v| v.as_f64());

            let allowed_local = fs_obj.get("enum")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect::<Vec<_>>());

            let allowed = allowed_local.or(None);

            out.push(FieldSpec {
                name: name.clone(),
                title,
                description,
                required: required.iter().any(|r| r == name),
                kind,
                default,
                min,
                max,
                allowed,
            });
        }

        return Ok(out);
    }

    bail!("no branch found for type={kind_key}");
}

fn discriminant_matches(props: &Map<String, Value>, kind_key: &str) -> bool {
    let Some(tval) = props.get("type") else {
        return false;
    };
    let Some(tobj) = tval.as_object() else {
        return false;
    };

    if tobj.get("const").and_then(|v| v.as_str()) == Some(kind_key) {
        return true;
    }
    if let Some(arr) = tobj.get("enum").and_then(|v| v.as_array()) {
        if arr.len() == 1 && arr[0].as_str() == Some(kind_key) {
            return true;
        }
    }
    false
}

/// Resolve a local $ref like "#/$defs/SeaParameters" against the root object.
/// Returns the referenced object map, or None if it can't be resolved.
fn resolve_ref_obj<'a>(
    root_obj: &'a Map<String, Value>,
    obj: &'a Map<String, Value>,
) -> Option<&'a Map<String, Value>> {
    match obj.get("$ref") {
        Some(Value::String(r)) => {
            let path = r.strip_prefix("#/")?;
            let mut cur: &Map<String, Value> = root_obj;
            for raw_seg in path.split('/') {
                // JSON Pointer unescape (~1 => /, ~0 => ~)
                let seg = raw_seg.replace("~1", "/").replace("~0", "~");
                cur = cur.get(&seg)?.as_object()?;
            }
            Some(cur)
        }
        _ => Some(obj),
    }
}

fn detect_field_kind(ty: Option<&Value>) -> Option<FieldKind> {
    match ty {
        Some(Value::String(s)) => match s.as_str() {
            "string" => Some(FieldKind::String),
            "integer" => Some(FieldKind::Integer),
            "number" => Some(FieldKind::Number),
            "boolean" => Some(FieldKind::Boolean),
            _ => None,
        },
        Some(Value::Array(arr)) => {
            // handle unions like ["null","integer"] for Option<T>
            arr.iter().filter_map(|v| v.as_str()).find_map(|s| match s {
                "string" => Some(FieldKind::String),
                "integer" => Some(FieldKind::Integer),
                "number" => Some(FieldKind::Number),
                "boolean" => Some(FieldKind::Boolean),
                "null" => None,
                _ => None,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::types::choices::{NoParams, StreamChoice};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
    #[serde(tag = "type", content = "params", rename_all = "kebab-case")]
    enum TinyChoice {
        Unit(NoParams),
    }

    #[test]
    fn detect_field_kind_handles_all_primitives() {
        assert!(matches!(
            detect_field_kind(Some(&Value::String("string".into()))),
            Some(FieldKind::String)
        ));
        assert!(matches!(
            detect_field_kind(Some(&Value::String("integer".into()))),
            Some(FieldKind::Integer)
        ));
        assert!(matches!(
            detect_field_kind(Some(&Value::String("number".into()))),
            Some(FieldKind::Number)
        ));
        assert!(matches!(
            detect_field_kind(Some(&Value::String("boolean".into()))),
            Some(FieldKind::Boolean)
        ));
        assert!(detect_field_kind(Some(&Value::String("object".into()))).is_none());
        assert!(detect_field_kind(None).is_none());
    }

    #[test]
    fn detect_field_kind_handles_nullable_union() {
        let a = Value::Array(vec![
            Value::String("null".into()),
            Value::String("integer".into()),
        ]);
        let b = Value::Array(vec![
            Value::String("integer".into()),
            Value::String("null".into()),
        ]);
        assert!(matches!(
            detect_field_kind(Some(&a)),
            Some(FieldKind::Integer)
        ));
        assert!(matches!(
            detect_field_kind(Some(&b)),
            Some(FieldKind::Integer)
        ));
    }

    #[test]
    fn resolve_ref_obj_direct_returns_self() {
        let root = json!({"$defs": {}}).as_object().unwrap().clone();
        let me = json!({"type":"integer"}).as_object().unwrap().clone();
        let out = resolve_ref_obj(&root, &me).unwrap();
        assert_eq!(out.get("type").and_then(Value::as_str), Some("integer"));
    }

    #[test]
    fn resolve_ref_obj_follow_refs_and_unescape() {
        let root = json!({
            "$defs": {
                "a~b": { "inner/seg": { "type": "number" } }
            }
        })
        .as_object()
        .unwrap()
        .clone();

        // ref to "#/$defs/a~0b/inner~1seg" (escaped "~" and "/")
        let obj = json!({ "$ref": "#/$defs/a~0b/inner~1seg" })
            .as_object()
            .unwrap()
            .clone();

        let out = resolve_ref_obj(&root, &obj).expect("ref resolved");
        assert_eq!(out.get("type").and_then(Value::as_str), Some("number"));
    }

    #[test]
    fn discriminant_matches_via_const_and_enum() {
        let props_const = json!({
            "type": { "const": "sea-generator" }
        })
        .as_object()
        .unwrap()
        .clone();

        let props_enum = json!({
            "type": { "enum": ["sea-generator"] }
        })
        .as_object()
        .unwrap()
        .clone();

        assert!(discriminant_matches(&props_const, "sea-generator"));
        assert!(discriminant_matches(&props_enum, "sea-generator"));
        assert!(!discriminant_matches(&props_const, "other"));
    }

    #[test]
    fn specs_for_kind_on_variant_with_no_params_is_empty() {
        let root = super::schema_for::<TinyChoice>();
        let v = specs_for_kind(&root, "unit").expect("ok");
        assert!(v.is_empty());
    }

    #[test]
    fn specs_for_kind_kind_not_found_errors() {
        let root = super::schema_for::<StreamChoice>();
        let err = specs_for_kind(&root, "does-not-exist").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no branch found"), "msg was: {msg}");
    }

    #[test]
    fn schema_for_wrapper_returns_object_like_schema() {
        let sch = super::schema_for::<StreamChoice>();
        let obj = sch.as_object().expect("root object");
        assert!(obj.contains_key("oneOf") || obj.contains_key("anyOf"));
    }
}
