function showCustomLoader() {
  const loader = document.getElementById('custom-loader');
  if (loader) loader.style.display = 'flex';
}

function hideCustomLoader() {
  const loader = document.getElementById('custom-loader');
  if (loader) loader.style.display = 'none';
}
