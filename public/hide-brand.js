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

// Microsoft SSO full-logout interceptor via fetch patching.
// Chainlit's frontend calls POST /logout via fetch, then navigates to /login.
// We intercept that fetch call, wait for it to finish (cookie cleared),
// then redirect to Microsoft's real logout before React can go to /login.
(function () {
  var MS_LOGOUT_URL =
    'https://login.microsoftonline.com/common/oauth2/v2.0/logout' +
    '?post_logout_redirect_uri=' +
    encodeURIComponent(window.location.origin);

  var _originalFetch = window.fetch;

  window.fetch = function () {
    var args = Array.prototype.slice.call(arguments);
    var url = args[0];
    var opts = args[1] || {};

    // Detect Chainlit's logout call
    var isLogout =
      (typeof url === 'string' && url.indexOf('/logout') !== -1) ||
      (url && url.url && url.url.indexOf('/logout') !== -1);
    var isPost =
      !opts.method ||
      opts.method.toUpperCase() === 'POST' ||
      opts.method.toUpperCase() === 'DELETE';

    if (isLogout && isPost) {
      return _originalFetch.apply(this, args).then(function (response) {
        // Cookie is now cleared by Chainlit — redirect to Microsoft logout
        window.location.href = MS_LOGOUT_URL;
        return response;
      }).catch(function () {
        // Even if the call fails, do the logout redirect
        window.location.href = MS_LOGOUT_URL;
      });
    }

    return _originalFetch.apply(this, args);
  };
})();

