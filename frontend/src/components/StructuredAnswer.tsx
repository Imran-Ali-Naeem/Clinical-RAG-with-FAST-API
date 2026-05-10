import type { ReactNode } from "react";

/** Split on a line boundary before each Markdown-style section heading `**Label:**`. */
function splitIntoSections(raw: string): string[] {
  const t = raw.trim();
  if (!t) return [];
  const parts = t.split(/\n(?=\*\*[^*\n]+\*\*:)/);
  return parts.map((p) => p.trim()).filter(Boolean);
}

/** Body after `**Title:**` on first line — rest can be prose and/or '-' bullets. */
function parseSectionBody(body: string): ReactNode[] {
  const lines = body.split("\n").map((l) => l.trimEnd());
  const out: ReactNode[] = [];
  let i = 0;

  const flushParagraph = (buf: string[]) => {
    if (!buf.length) return;
    const text = buf.join(" ").trim();
    if (text) out.push(<p key={out.length}>{text}</p>);
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
            <li key={idx}>{item}</li>
          ))}
        </ul>
      );
      continue;
    }
    paraBuf.push(line);
    i++;
  }
  flushParagraph(paraBuf);

  return out.length ? out : [<p key={0}>{body.trim()}</p>];
}

function parseSection(chunk: string, index: number): ReactNode {
  const headerPrefix = /^\*\*([^*:]+):\*\*\s*/;
  const m = chunk.match(headerPrefix);
  if (!m) {
    return (
      <div key={index} className="answer-md-section answer-md-plain">
        {chunk}
      </div>
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

/** Renders structured clinical answers without pulling in a Markdown dependency. */
export function StructuredAnswer({ text }: StructuredAnswerProps) {
  const sections = splitIntoSections(text);
  if (sections.length === 0) return null;
  if (sections.length === 1 && !sections[0].includes("**")) {
    return <div className="answer-md-fallback">{text}</div>;
  }

  return (
    <div className="answer-markdown">{sections.map((chunk, i) => parseSection(chunk, i))}</div>
  );
}
