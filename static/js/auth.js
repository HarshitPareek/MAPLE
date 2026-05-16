// Auth state management
const Auth = {
  setToken(token, user) {
    localStorage.setItem('cg_token', token);
    localStorage.setItem('cg_user', JSON.stringify(user));
    this._updateNav();
  },
  getToken()   { return localStorage.getItem('cg_token'); },
  getUser()    { try { return JSON.parse(localStorage.getItem('cg_user')); } catch { return null; } },
  isLoggedIn() { return !!this.getToken(); },
  logout() {
    localStorage.removeItem('cg_token');
    localStorage.removeItem('cg_user');
    this._updateNav();
    window.location.href = '/';
  },

  _updateNav() {
    const loggedIn = this.isLoggedIn();
    const authBtn    = document.getElementById('nav-auth');
    const logoutBtn  = document.getElementById('nav-logout');
    const profileLnk = document.getElementById('nav-profile');
    const watchLnk   = document.getElementById('nav-watchlist');
    const navDot     = document.querySelector('.nav-dot');
    if (authBtn)    authBtn.classList.toggle('hidden', loggedIn);
    if (logoutBtn)  logoutBtn.classList.toggle('hidden', !loggedIn);
    if (profileLnk) profileLnk.classList.toggle('hidden', !loggedIn);
    if (watchLnk)   watchLnk.classList.toggle('hidden', !loggedIn);
    if (navDot)     navDot.classList.toggle('hidden', !loggedIn);
  }
};

// Wire up logout button
document.addEventListener('DOMContentLoaded', () => {
  Auth._updateNav();
  const logoutBtn = document.getElementById('nav-logout');
  if (logoutBtn) logoutBtn.addEventListener('click', () => Auth.logout());
});
