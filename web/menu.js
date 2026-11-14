const status = document.getElementById("menu-status");
try {
  const response = await fetch("/api/meta");
  if (response.status === 503) {
    status.textContent = "No processed posts yet. Run python -m detector all --posts 10000, then restart the server.";
  } else if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  } else {
    const meta = await response.json();
    status.textContent = [
      `${meta.n_posts.toLocaleString()} posts`,
      `${meta.n_clusters} groups`,
      `${meta.n_clusters - meta.n_discarded} shown`,
      meta.chat.available ? `chat: ${meta.chat.model}` : "chat unavailable",
    ].join(" · ");
    if (meta.dataset?.burstiness_reliable === false) status.textContent += " · snapshot too short for burst flags";
  }
} catch (error) {
  status.textContent = `Could not reach the server (${error.message}).`;
}
