use crate::core::attributes::{AttributeRef, NominalAttribute, NumericAttribute};
use crate::core::instance_header::InstanceHeader;
use crate::utils::file_parsing::{split_csv_preserving_quotes, strip_surrounding_quotes};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind, Seek};
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum AttributeKind {
    Numeric,
    Nominal(Vec<String>),
}

pub(super) fn is_comment_or_empty(s: &str) -> bool {
    let t = s.trim();
    t.is_empty() || t.starts_with('%')
}

pub(super) fn parse_header(
    reader: &mut BufReader<File>,
    class_index: usize,
) -> Result<(InstanceHeader, u64), Error> {
    let mut relation: Option<String> = None;
    let mut attributes: Vec<AttributeRef> = Vec::new();
    let mut line = String::new();
    let mut pending_line: Option<String> = None;

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "ARFF file ended before @data",
            ));
        }
        if is_comment_or_empty(&line) {
            continue;
        }

        let low = line.to_lowercase();
        if low.starts_with("@relation") {
            let raw = line.trim()[9..].trim();
            let rel = strip_surrounding_quotes(raw).to_string();
            relation = Some(rel);
            break;
        } else if low.starts_with("@attribute") || low.starts_with("@data") {
            pending_line = Some(line.clone());
            break;
        }
    }

    let data_start_pos: u64;
    loop {
        if let Some(pending) = pending_line.take() {
            line = pending;
        } else {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                return Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    "ARFF file ended before @data",
                ));
            }
        }

        if is_comment_or_empty(&line) {
            continue;
        }

        let low = line.to_lowercase();
        if low.starts_with("@attribute") {
            let (name, kind) = parse_attribute_line(&line)?;
            match kind {
                AttributeKind::Numeric => {
                    let attribute = NumericAttribute::new(name);
                    attributes.push(Arc::new(attribute) as AttributeRef);
                }
                AttributeKind::Nominal(values) => {
                    let mut map = HashMap::new();
                    for (i, v) in values.iter().enumerate() {
                        map.insert(v.clone(), i);
                    }
                    let attribute = NominalAttribute::with_values(name, values, map);
                    attributes.push(Arc::new(attribute) as AttributeRef);
                }
            }
        } else if low.starts_with("@data") {
            data_start_pos = reader.stream_position()?;
            break;
        } else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Unsupported header directive: {}", line.trim()),
            ));
        }
    }

    let header = InstanceHeader::new(
        relation.unwrap_or_else(|| "unnamed_relation".to_string()),
        attributes,
        class_index,
    );

    Ok((header, data_start_pos))
}

pub(super) fn parse_attribute_line(line: &str) -> Result<(String, AttributeKind), Error> {
    let rest = {
        let mut l = line.trim();
        let low = l.to_ascii_lowercase();
        if !low.starts_with("@attribute") {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Line is not '@attribute'",
            ));
        }
        if let Some(idx) = low.find("@attribute") {
            l = &l[idx + "@attribute".len()..];
        }
        l.trim()
    };

    let (name, after_name) = if rest.starts_with('\'') || rest.starts_with('"') {
        let quote = rest.chars().next().unwrap();
        let mut end = None;
        for (i, c) in rest.char_indices().skip(1) {
            if c == quote {
                end = Some(i);
                break;
            }
        }
        let end = end.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                "Attribute name without closing quote marks",
            )
        })?;
        let name = rest[1..end].to_string();
        (name, rest[end + 1..].trim())
    } else {
        let mut it = rest.splitn(2, char::is_whitespace);
        let name = it.next().unwrap().to_string();
        let after = it
            .next()
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Attribute type is missing"))?;
        (name, after.trim())
    };

    let low = after_name.to_ascii_lowercase();
    if low.starts_with("numeric") || low.starts_with("real") || low.starts_with("integer") {
        return Ok((name, AttributeKind::Numeric));
    }

    let after_name = after_name.trim();
    if after_name.starts_with('{') {
        let close = after_name
            .rfind('}')
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Nominal set without closing '}'"))?;

        let inside = &after_name[1..close];
        let values = inside
            .split(',')
            .map(|s| strip_surrounding_quotes(s.trim()).to_string())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        if values.is_empty() {
            return Err(Error::new(ErrorKind::InvalidData, "Empty nominal domain"));
        }

        return Ok((name, AttributeKind::Nominal(values)));
    }

    Err(Error::new(
        ErrorKind::InvalidData,
        format!("Attribute kind not supported: {after_name}"),
    ))
}

pub(super) fn parse_instance_values(
    header: &InstanceHeader,
    line: &str,
) -> Result<Vec<f64>, Error> {
    let tokens = split_csv_preserving_quotes(line);
    if tokens.len() != header.attributes.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "Number of columns ({}) differs from number of attributes ({})",
                tokens.len(),
                header.attributes.len()
            ),
        ));
    }

    let mut values = Vec::with_capacity(tokens.len());
    for (idx, raw) in tokens.into_iter().enumerate() {
        let raw = raw.trim();
        if raw == "?" {
            values.push(f64::NAN);
            continue;
        }

        let attr = &header.attributes[idx];

        if attr.as_any().is::<NumericAttribute>() {
            let v: f64 = raw.parse().map_err(|_| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid numeric value '{raw}' for attribute #{idx}"),
                )
            })?;
            values.push(v);
            continue;
        }

        if let Some(nominal) = attr.as_any().downcast_ref::<NominalAttribute>() {
            let key = strip_surrounding_quotes(raw);
            let Some(&pos) = nominal.label_to_index.get(key) else {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Nominal value '{key}' not found in domain of attribute #{idx}"),
                ));
            };
            values.push(pos as f64);
            continue;
        }

        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Unsupported attribute type at column #{idx}"),
        ));
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::attributes::{Attribute, AttributeRef, NominalAttribute, NumericAttribute};
    use crate::core::instance_header::InstanceHeader;
    use std::any::Any;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufReader, ErrorKind, Write};
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn hdr(attrs: Vec<AttributeRef>, class_index: usize) -> InstanceHeader {
        InstanceHeader::new("r".into(), attrs, class_index)
    }

    fn write_temp(contents: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(contents.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_attribute_line_missing_type_after_name() {
        let err = parse_attribute_line("@attribute outlook").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_attribute_line_name_without_closing_quote() {
        let err = parse_attribute_line("@attribute 'bad {x, y}").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_attribute_line_rejects_non_attribute_line() {
        let err = parse_attribute_line("@relation r").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_attribute_line_empty_nominal_domain() {
        let err = parse_attribute_line("@attribute a {}").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_attribute_line_whitespace_only_nominal_domain() {
        let err = parse_attribute_line("@attribute a {   }").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_attribute_line_trailing_comma_nominal_domain() {
        let (name, kind) = parse_attribute_line("@attribute a {x, }").unwrap();
        assert_eq!(name, "a");
        match kind {
            AttributeKind::Nominal(v) => assert_eq!(v, vec!["x"]),
            _ => panic!("expected nominal"),
        }
    }

    #[test]
    fn parse_attribute_line_nominal_missing_closing_brace() {
        let err = parse_attribute_line("@attribute a {x, y").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_attribute_line_unsupported_type_string() {
        let err = parse_attribute_line("@attribute note string").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_instance_values_wrong_arity() {
        let h = hdr(
            vec![
                Arc::new(NumericAttribute::new("a".into())) as AttributeRef,
                Arc::new(NumericAttribute::new("b".into())) as AttributeRef,
            ],
            0,
        );
        let err = parse_instance_values(&h, "1").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_instance_values_invalid_numeric() {
        let h = hdr(
            vec![Arc::new(NumericAttribute::new("x".into())) as AttributeRef],
            0,
        );
        let err = parse_instance_values(&h, "abc").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_instance_values_unknown_nominal() {
        let values = vec!["x".into(), "y".into()];
        let mut map = HashMap::new();
        map.insert("x".into(), 0);
        map.insert("y".into(), 1);
        let nom = NominalAttribute::with_values("a".into(), values, map);
        let h = hdr(vec![Arc::new(nom) as AttributeRef], 0);
        let err = parse_instance_values(&h, "z").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[derive(Debug)]
    struct DummyAttr;
    impl Attribute for DummyAttr {
        fn name(&self) -> String {
            "d".into()
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn arff_representation(&self) -> String {
            "@attribute d dummy".into()
        }

        fn calc_memory_size(&self) -> usize {
            size_of::<Self>()
        }
    }

    #[test]
    fn parse_instance_values_unsupported_attribute_type() {
        let h = hdr(vec![Arc::new(DummyAttr) as AttributeRef], 0);
        let err = parse_instance_values(&h, "42").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_header_unexpected_eof_before_data() {
        let tf = write_temp("@relation r\n@attribute a numeric\n");
        let mut br = BufReader::new(File::open(tf.path()).unwrap());
        let err = parse_header(&mut br, 0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
    }

    #[test]
    fn parse_header_unsupported_header_directive() {
        let tf = write_temp("@relation r\n@foo bar\n@data\n1\n");
        let mut br = BufReader::new(File::open(tf.path()).unwrap());
        let err = parse_header(&mut br, 0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn parse_header_attribute_before_relation_is_reprocessed() {
        let tf = write_temp("@attribute a numeric\n@data\n1\n");
        let mut br = BufReader::new(File::open(tf.path()).unwrap());
        let (h, _pos) = parse_header(&mut br, 0).unwrap();
        assert_eq!(h.relation_name(), "unnamed_relation");
        assert_eq!(h.number_of_attributes(), 1);
    }
}
