export function groupLabel(value, id) {
  const label = String(value || "").replace(/^#{1,6}\s+/, "").trim();
  return label || `Group ${id}`;
}
