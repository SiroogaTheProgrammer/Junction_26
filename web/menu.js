const status = document.getElementById("menu-status");
try {
  const response = await fetch("/api/live/chat/status");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const { chat, window_posts } = await response.json();
  status.textContent = `${window_posts.toLocaleString()} posts in live window · ${chat.available ? "chat configured" : chat.error}`;
} catch (error) {
  status.textContent = `Could not reach the server (${error.message}).`;
}
