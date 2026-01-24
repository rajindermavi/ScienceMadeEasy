# Schemas

## extract_details.json

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "extract_details.schema.json",
  "title": "ExtractDetails",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "arviv_query",
    "papers",
    "paper_index",
    "md_details",
    "txt_details",
    "md_bm25",
    "md_qdrant",
    "txt_bm25",
    "txt_qdrant"
  ],
  "properties": {
    "arviv_query": { "type": "string" },
    "papers": {
      "type": "object",
      "additionalProperties": { "$ref": "#/definitions/PaperRecord" }
    },
    "paper_index": {
      "type": "array",
      "items": { "type": "string" }
    },
    "md_details": { "$ref": "#/definitions/MdDetails" },
    "txt_details": { "$ref": "#/definitions/TxtDetails" },
    "md_bm25": { "$ref": "#/definitions/WhooshIndexSummary" },
    "txt_bm25": { "$ref": "#/definitions/WhooshIndexSummary" },
    "md_qdrant": { "$ref": "#/definitions/QdrantIndexSummary" },
    "txt_qdrant": { "$ref": "#/definitions/QdrantIndexSummary" }
  },
  "definitions": {
    "PaperRecord": {
      "type": "object",
      "additionalProperties": false,
      "required": ["meta", "extract"],
      "properties": {
        "meta": { "$ref": "#/definitions/PaperMeta" },
        "extract": { "type": "boolean" },
        "latex_dir": { "type": ["string", "null"] },
        "combined_latex_path": { "type": ["string", "null"] },
        "md_full_path": { "type": ["string", "null"] },
        "txt_full_path": { "type": ["string", "null"] },
        "md_chunk_files": { "type": ["string", "null"] },
        "txt_chunk_files": { "type": ["string", "null"] }
      }
    },
    "PaperMeta": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "arxiv_id",
        "base_id",
        "sanitized_id",
        "version",
        "title",
        "primary_category",
        "categories",
        "authors",
        "published_date",
        "updated_date",
        "url",
        "extract",
        "citation_list",
        "reference_list"
      ],
      "properties": {
        "arxiv_id": { "type": "string" },
        "base_id": { "type": "string" },
        "sanitized_id": { "type": "string" },
        "version": { "type": "string" },
        "title": { "type": "string" },
        "primary_category": { "type": "string" },
        "categories": { "type": "array", "items": { "type": "string" } },
        "authors": { "type": "array", "items": { "type": "string" } },
        "published_date": { "type": "string" },
        "updated_date": { "type": "string" },
        "url": { "type": "string" },
        "gzip": { "type": ["string", "null"] },
        "summary": { "type": ["string", "null"] },
        "comment": { "type": ["string", "null"] },
        "extract": { "type": ["boolean", "null"] },
        "citation_list": {
          "type": ["array", "null"],
          "items": { "type": ["string", "null"] }
        },
        "reference_list": {
          "type": ["array", "null"],
          "items": { "type": ["string", "null"] }
        }
      }
    },
    "MdDetails": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "md_json_files",
        "records_seen",
        "records_written",
        "unique_paper_ids",
        "output_jsonl",
        "normalized_records"
      ],
      "properties": {
        "md_json_files": { "type": "integer" },
        "records_seen": { "type": "integer" },
        "records_written": { "type": "integer" },
        "unique_paper_ids": { "type": "array", "items": { "type": "string" } },
        "output_jsonl": { "type": "string" },
        "normalized_records": {
          "type": "array",
          "items": { "$ref": "#/definitions/MdChunkRecord" }
        }
      }
    },
    "MdChunkRecord": {
      "type": "object",
      "additionalProperties": true,
      "required": [
        "chunk_id",
        "paper_id",
        "source_file",
        "section",
        "labels",
        "refs",
        "neighbors",
        "start_line",
        "end_line",
        "text",
        "text_len",
        "token_estimate",
        "equations_raw",
        "equation_count",
        "added_at",
        "version"
      ],
      "properties": {
        "chunk_id": { "type": "string" },
        "paper_id": { "type": "string" },
        "source_file": { "type": "string" },
        "section": { "type": "string" },
        "labels": { "type": "array", "items": { "type": "string" } },
        "refs": { "type": "array", "items": { "type": "string" } },
        "neighbors": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["id", "direction"],
            "properties": {
              "id": { "type": "string" },
              "direction": { "type": "string" }
            }
          }
        },
        "start_line": { "type": ["integer", "null"] },
        "end_line": { "type": ["integer", "null"] },
        "text": { "type": "string" },
        "text_len": { "type": "integer" },
        "token_estimate": { "type": "integer" },
        "equations_raw": { "type": "array", "items": { "type": "string" } },
        "equation_count": { "type": "integer" },
        "added_at": { "type": "integer" },
        "version": { "type": "string" }
      }
    },
    "TxtDetails": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "input_files",
        "records_seen",
        "records_written",
        "unique_paper_ids",
        "output_jsonl",
        "normalized_records"
      ],
      "properties": {
        "input_files": { "type": "integer" },
        "records_seen": { "type": "integer" },
        "records_written": { "type": "integer" },
        "unique_paper_ids": { "type": "array", "items": { "type": "string" } },
        "output_jsonl": { "type": "string" },
        "normalized_records": {
          "type": "array",
          "items": { "$ref": "#/definitions/TxtChunkRecord" }
        }
      }
    },
    "TxtChunkRecord": {
      "type": "object",
      "additionalProperties": true,
      "required": [
        "chunk_id",
        "paper_id",
        "source_file",
        "section_path",
        "section",
        "start_line",
        "end_line",
        "chunk_type",
        "text",
        "text_len",
        "token_estimate",
        "has_math_loss",
        "labels",
        "neighbors",
        "meta",
        "harvest",
        "added_at",
        "version"
      ],
      "properties": {
        "chunk_id": { "type": "string" },
        "paper_id": { "type": "string" },
        "source_file": { "type": "string" },
        "section_path": { "type": "string" },
        "section": { "type": "string" },
        "start_line": { "type": ["integer", "null"] },
        "end_line": { "type": ["integer", "null"] },
        "chunk_type": { "type": "string" },
        "text": { "type": "string" },
        "text_len": { "type": "integer" },
        "token_estimate": { "type": "integer" },
        "has_math_loss": { "type": "boolean" },
        "labels": { "type": "array", "items": { "type": "string" } },
        "neighbors": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["id", "direction"],
            "properties": {
              "id": { "type": "string" },
              "direction": { "type": "string" }
            }
          }
        },
        "meta": { "type": "object" },
        "harvest": {
          "type": "object",
          "additionalProperties": false,
          "required": ["arxiv_ids", "emails", "urls"],
          "properties": {
            "arxiv_ids": { "type": "array", "items": { "type": "string" } },
            "emails": { "type": "array", "items": { "type": "string" } },
            "urls": { "type": "array", "items": { "type": "string" } }
          }
        },
        "added_at": { "type": "integer" },
        "version": { "type": "string" }
      }
    },
    "WhooshIndexSummary": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "records_indexed",
        "skipped_blank",
        "skipped_decode",
        "skipped_filtered",
        "index_path",
        "description"
      ],
      "properties": {
        "records_indexed": { "type": "integer" },
        "skipped_blank": { "type": "integer" },
        "skipped_decode": { "type": "integer" },
        "skipped_filtered": { "type": "integer" },
        "index_path": { "type": "string" },
        "description": { "type": "string" }
      }
    },
    "QdrantIndexSummary": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "records_indexed",
        "skipped_blank",
        "skipped_decode",
        "skipped_empty_text",
        "collection_name",
        "embedding_model",
        "index_path"
      ],
      "properties": {
        "records_indexed": { "type": "integer" },
        "skipped_blank": { "type": "integer" },
        "skipped_decode": { "type": "integer" },
        "skipped_empty_text": { "type": "integer" },
        "collection_name": { "type": "string" },
        "embedding_model": { "type": "string" },
        "index_path": { "type": "string" }
      }
    }
  }
}
```
