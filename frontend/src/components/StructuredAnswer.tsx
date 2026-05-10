import type { ReactNode } from "react";

/** Split before each `**Section label:**` heading (newline not required — models often glue headings to prior text). */
function splitIntoSections(raw: string): string[] {
  const t = raw.trim();
  if (!t) return [];
  const parts = t.split(/(?=\*\*[^*\n]+:\*\*)/).map((p) => p.trim());
  return parts.filter(Boolean);
}

/** Convert `* ` list markers (and inline ` * ` sequences) to `- ` so the line parser can build <ul>. */
function normalizeBulletMarkers(body: string): string {
  let s = body.trim();
  // First token is a bullet: "* Item..."
  s = s.replace(/^\*\s+/, "- ");
  // Space-separated bullets mid-line: "foo * Next item"
  s = s.replace(/\s+\*\s+/g, "\n- ");
  // Any line that starts with "* " after newline
  s = s.replace(/^\*\s+/gm, "- ");
  return s;
}

/** Turn `**bold**` into React nodes (LLM output stays visible otherwise). */
function renderInlineFormatting(text: string, keyPrefix: string): ReactNode {
  const segments = text.split(/(\*\*[^*]+\*\*)/g);
  return segments.map((seg, i) => {
    if (seg.startsWith("**") && seg.endsWith("**") && seg.length > 4) {
      return (
        <strong key={`${keyPrefix}-${i}`}>
          {seg.slice(2, -2)}
        </strong>
      );
    }
    return <span key={`${keyPrefix}-${i}`}>{seg}</span>;
  });
}

function parseSectionBody(body: string): ReactNode[] {
  const normalized = normalizeBulletMarkers(body);
  const lines = normalized.split("\n").map((l) => l.trimEnd());
  const out: ReactNode[] = [];
  let i = 0;

  const flushParagraph = (buf: string[]) => {
    if (!buf.length) return;
    const text = buf.join(" ").trim();
    if (text) {
      out.push(
        <p key={out.length}>{renderInlineFormatting(text, `p-${out.length}`)}</p>
      );
    }
  };

  const paraBuf: string[] = [];
  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line) {
      i++;
      continue;
    }
    if (line.startsWith("- ")) {
      flushParagraph(paraBuf);
      paraBuf.length = 0;
      const items: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("- ")) {
        items.push(lines[i].trim().replace(/^-\s+/, ""));
        i++;
      }
      out.push(
        <ul key={out.length} className="answer-md-list">
          {items.map((item, idx) => (
            <li key={idx}>{renderInlineFormatting(item, `li-${out.length}-${idx}`)}</li>
          ))}
        </ul>
      );
      continue;
    }
    paraBuf.push(line);
    i++;
  }
  flushParagraph(paraBuf);

  return out.length ? out : [<p key={0}>{renderInlineFormatting(body.trim(), "p0")}</p>];
}

function parseSection(chunk: string, index: number): ReactNode {
  const headerPrefix = /^\*\*([^*:]+):\*\*\s*/;
  const m = chunk.match(headerPrefix);
  if (!m) {
    const bodyPieces = parseSectionBody(chunk);
    return (
      <section key={index} className="answer-md-section answer-md-preamble">
        <div className="answer-md-body">{bodyPieces}</div>
      </section>
    );
  }
  const title = m[1].trim();
  const rest = chunk.slice(m[0].length).trimEnd();
  const bodyPieces = parseSectionBody(rest || "");

  return (
    <section key={index} className="answer-md-section">
      <h3 className="answer-md-heading">{title}</h3>
      <div className="answer-md-body">{bodyPieces}</div>
    </section>
  );
}

interface StructuredAnswerProps {
  /** Full model output with `**Summary:**` etc. */
  text: string;
}

/** Renders clinical answers: headings, lists (`-` or `*`), and inline **bold** without a Markdown library. */
export function StructuredAnswer({ text }: StructuredAnswerProps) {
  const sections = splitIntoSections(text);
  if (sections.length === 0) return null;

  // Only use plain fallback when there is no Markdown structure at all
  if (sections.length === 1 && !sections[0].includes("**")) {
    const bodyPieces = parseSectionBody(sections[0]);
    return (
      <div className="answer-markdown answer-md-single">{bodyPieces}</div>
    );
  }

  return (
    <div className="answer-markdown">{sections.map((chunk, i) => parseSection(chunk, i))}</div>
  );
}
