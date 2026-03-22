// Minimal, safe customization: remove watermark and outbound Chainlit links once.
(function () {
  function cleanOnce() {
    document
      .querySelectorAll('.watermark, a[href*="chainlit.io" i]')
      .forEach((el) => el.remove());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', cleanOnce);
  } else {
    cleanOnce();
  }
})();
