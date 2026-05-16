// Movie card rendering + grid/row + detail panel + pagination
let userFavMovies     = [], userFavTV     = [];
let userWatchedMovies = [], userWatchedTV = [];
let userLikedMovies   = [], userLikedTV   = [];
let userWatchlistMovies = [], userWatchlistTV = [];
let userNIMovies      = [], userNITV      = [];

async function loadUserLists() {
  if (!Auth.isLoggedIn()) return;
  try {
    const [fData, wData, lData, wlData, niData] = await Promise.all([
      API.get('/api/user/favorites'),
      API.get('/api/user/watched'),
      API.get('/api/user/liked'),
      API.get('/api/user/watchlist'),
      API.get('/api/user/not-interested'),
    ]);
    userFavMovies       = fData.movie_ids  || [];
    userFavTV           = fData.tv_ids     || [];
    userWatchedMovies   = wData.movie_ids  || [];
    userWatchedTV       = wData.tv_ids     || [];
    userLikedMovies     = lData.movie_ids  || [];
    userLikedTV         = lData.tv_ids     || [];
    userWatchlistMovies = wlData.movie_ids || [];
    userWatchlistTV     = wlData.tv_ids    || [];
    userNIMovies        = niData.movie_ids || [];
    userNITV            = niData.tv_ids    || [];
  } catch {}
}

function _isFav(id, ct)       { return ct === 'tv' ? userFavTV.includes(id)       : userFavMovies.includes(id); }
function _isWatched(id, ct)   { return ct === 'tv' ? userWatchedTV.includes(id)   : userWatchedMovies.includes(id); }
function _isLiked(id, ct)     { return ct === 'tv' ? userLikedTV.includes(id)     : userLikedMovies.includes(id); }
function _isWatchlist(id, ct) { return ct === 'tv' ? userWatchlistTV.includes(id) : userWatchlistMovies.includes(id); }
function _isNI(id, ct)        { return ct === 'tv' ? userNITV.includes(id)        : userNIMovies.includes(id); }

function makeCard(movie) {
  const poster = movie.poster || '/static/assets/placeholder.png';
  const ct     = movie.content_type || 'movie';
  const isFav       = _isFav(movie.id, ct);
  const isLiked     = _isLiked(movie.id, ct);
  const isWatchlist = _isWatchlist(movie.id, ct);

  const tvBadge = ct === 'tv'
    ? `<span style="font-size:0.7rem;background:rgba(99,102,241,0.35);border:1px solid rgba(99,102,241,0.5);border-radius:4px;padding:1px 6px;color:#a5b4fc;">TV</span>`
    : '';

  const safeTitle = movie.title.replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  const card = document.createElement('div');
  card.className = 'movie-card';
  card.dataset.id = movie.id;
  card.dataset.ct  = ct;
  card.innerHTML = `
    <div class="card-bg" style="background-image:url('${poster}')"></div>
    <img src="${poster}" alt="${safeTitle}" loading="lazy" onerror="this.src='/static/assets/placeholder.png'">
    <div class="movie-card-overlay">
      <div class="movie-card-title">${safeTitle}</div>
      <div class="movie-card-meta">
        ${movie.year ? `<span>${movie.year}</span>` : ''}
        <span class="rating-badge">★ ${movie.rating}</span>
        ${tvBadge}
      </div>
      ${Auth.isLoggedIn() ? `
      <div class="movie-card-actions">
        <button class="icon-btn like-btn ${isLiked ? 'active' : ''}" title="Like" data-id="${movie.id}" data-ct="${ct}">▲</button>
        <button class="icon-btn fav-btn ${isFav ? 'active' : ''}" title="Favorite" data-id="${movie.id}" data-ct="${ct}">♥</button>
        <button class="icon-btn watchlist-btn ${isWatchlist ? 'active' : ''}" title="Watchlist" data-id="${movie.id}" data-ct="${ct}">+</button>
        <button class="icon-btn ni-btn ${_isNI(movie.id, ct) ? 'active' : ''}" title="Not Interested" data-id="${movie.id}" data-ct="${ct}">✕</button>
      </div>` : ''}
    </div>
  `;

  card.addEventListener('click', (e) => {
    if (e.target.closest('.movie-card-actions')) return;
    openDetail(movie.id, ct);
  });

  // Like toggle
  card.querySelectorAll('.like-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = parseInt(btn.dataset.id), type = btn.dataset.ct;
      const active = btn.classList.contains('active');
      if (active) {
        await API.delete(`/api/user/liked/${id}?type=${type}`).catch(() => {});
        if (type === 'tv') userLikedTV     = userLikedTV.filter(x => x !== id);
        else               userLikedMovies = userLikedMovies.filter(x => x !== id);
        btn.classList.remove('active');
        Toast.success('Like removed');
      } else {
        await API.post('/api/user/liked', { movie_id: id, content_type: type }).catch(() => {});
        if (type === 'tv') userLikedTV.push(id);
        else               userLikedMovies.push(id);
        btn.classList.add('active');
        Toast.success('Liked! Updating your recommendations…');
      }
    });
  });

  // Favorite toggle
  card.querySelectorAll('.fav-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = parseInt(btn.dataset.id), type = btn.dataset.ct;
      const active = btn.classList.contains('active');
      if (active) {
        await API.delete(`/api/user/favorites/${id}?type=${type}`).catch(() => {});
        if (type === 'tv') userFavTV     = userFavTV.filter(x => x !== id);
        else               userFavMovies = userFavMovies.filter(x => x !== id);
        btn.classList.remove('active');
        Toast.success('Removed from favorites');
      } else {
        await API.post('/api/user/favorites', { movie_id: id, content_type: type }).catch(() => {});
        if (type === 'tv') userFavTV.push(id);
        else               userFavMovies.push(id);
        btn.classList.add('active');
        Toast.success('Added to favorites');
      }
    });
  });

  // Watched toggle
  card.querySelectorAll('.watched-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = parseInt(btn.dataset.id), type = btn.dataset.ct;
      const active = btn.classList.contains('active');
      if (active) {
        await API.delete(`/api/user/watched/${id}?type=${type}`).catch(() => {});
        if (type === 'tv') userWatchedTV     = userWatchedTV.filter(x => x !== id);
        else               userWatchedMovies = userWatchedMovies.filter(x => x !== id);
        btn.classList.remove('active');
        Toast.success('Removed from watched');
      } else {
        await API.post('/api/user/watched', { movie_id: id, content_type: type }).catch(() => {});
        if (type === 'tv') userWatchedTV.push(id);
        else               userWatchedMovies.push(id);
        btn.classList.add('active');
        Toast.success('Marked as watched');
      }
    });
  });

  // Watchlist toggle
  card.querySelectorAll('.watchlist-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = parseInt(btn.dataset.id), type = btn.dataset.ct;
      const active = btn.classList.contains('active');
      if (active) {
        await API.delete(`/api/user/watchlist/${id}?type=${type}`).catch(() => {});
        if (type === 'tv') userWatchlistTV     = userWatchlistTV.filter(x => x !== id);
        else               userWatchlistMovies = userWatchlistMovies.filter(x => x !== id);
        btn.classList.remove('active');
        Toast.success('Removed from watchlist');
      } else {
        await API.post('/api/user/watchlist', { movie_id: id, content_type: type }).catch(() => {});
        if (type === 'tv') userWatchlistTV.push(id);
        else               userWatchlistMovies.push(id);
        btn.classList.add('active');
        Toast.success('Added to watchlist');
      }
    });
  });

  // Not interested toggle
  card.querySelectorAll('.ni-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = parseInt(btn.dataset.id), type = btn.dataset.ct;
      const active = btn.classList.contains('active');
      if (active) {
        await API.delete(`/api/user/not-interested/${id}?type=${type}`).catch(() => {});
        if (type === 'tv') userNITV     = userNITV.filter(x => x !== id);
        else               userNIMovies = userNIMovies.filter(x => x !== id);
        btn.classList.remove('active');
        Toast.success('Removed from not interested');
      } else {
        await API.post('/api/user/not-interested', { movie_id: id, content_type: type }).catch(() => {});
        if (type === 'tv') userNITV.push(id);
        else               userNIMovies.push(id);
        btn.classList.add('active');
        // Fade out the card
        card.style.opacity = '0.3';
        card.style.pointerEvents = 'none';
        Toast.success('Not interested — we\'ll show less like this');
      }
    });
  });

  return card;
}

// ---- Bubble hover — TILE inflates like a bubble, neighbours deform ----
const BUBBLE_SCALE     = 1.12;
const BUBBLE_NUDGE     = 16; // max px neighbours are pushed
const BUBBLE_SQUISH    = 0.04; // max scale deformation on neighbours
const BUBBLE_RADIUS    = 1.8; // influence multiplier

function setupBubbleEffect(container) {
  const cards = [...container.querySelectorAll('.movie-card')];
  if (cards.length < 2) return;

  cards.forEach(card => {
    card.addEventListener('mouseenter', function () {
      // Inflate the hovered card like a bubble
      this.style.transform = `scale(${BUBBLE_SCALE})`;
      this.style.zIndex    = '20';

      const h   = this.getBoundingClientRect();
      const hCx = h.left + h.width  / 2;
      const hCy = h.top  + h.height / 2;
      const influenceR = Math.max(h.width, h.height) * BUBBLE_RADIUS;

      cards.forEach(neighbor => {
        if (neighbor === this) return;
        const r  = neighbor.getBoundingClientRect();
        const cx = r.left + r.width  / 2;
        const cy = r.top  + r.height / 2;
        const dx = cx - hCx, dy = cy - hCy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist === 0 || dist > influenceR) return;

        const t = 1 - dist / influenceR; // 0..1 proximity factor
        const pushX = (dx / dist) * t * BUBBLE_NUDGE;
        const pushY = (dy / dist) * t * BUBBLE_NUDGE;

        // Squish: scale down slightly when very close (bubble surface tension)
        const squish = 1 - t * BUBBLE_SQUISH;
        // Stretch axis: elongate slightly away from hovered card
        const angle  = Math.atan2(dy, dx) * (180 / Math.PI);

        neighbor.style.transform =
          `translate(${pushX}px, ${pushY}px) scale(${squish}) rotate(${angle}deg) scaleX(${1 + t * 0.02}) rotate(${-angle}deg)`;
        neighbor.classList.add('bubble-neighbor');
      });
    });

    card.addEventListener('mouseleave', function () {
      this.style.transform = '';
      this.style.zIndex    = '';
      cards.forEach(c => {
        if (c !== this) {
          c.style.transform = '';
          c.classList.remove('bubble-neighbor');
        }
      });
    });
  });
}

// ---- Square bento layout ----
// Only 1×1 (small) and 2×2 (big) square tiles — no rectangles.
//
// 6-card repeating block, 6 cols × 2 rows = 12 cells:
//
//   Pattern A: [B B][s][s][B B]    B: [B B][B B][s][s]    C: [s][s][B B][B B]
//              [B B][s][s][B B]       [B B][B B][s][s]       [s][s][B B][B B]
//
// Math: 2 big (4 cells each = 8) + 4 small (1 cell each = 4) = 12 ✓
// 24 cards ÷ 6 = 4 exact groups, 0 orphans.

function _bentoColCount() {
  const w = window.innerWidth;
  if (w <= 560) return 2;
  if (w <= 900) return 4;
  return 6;
}

// Set grid-auto-rows so tiles have 2:3 portrait aspect ratio (like movie posters).
// row height = column width × 1.5 → small tile = 1×1 (2:3), big tile = 2×2 (2:3).
// Retries via rAF if container hasn't painted yet (clientWidth === 0).
function setSquareRows(container) {
  const cols  = _bentoColCount();
  const gapPx = parseFloat(getComputedStyle(container).columnGap) || 8;
  const w     = container.clientWidth;
  if (w === 0) { requestAnimationFrame(() => setSquareRows(container)); return; }
  const cellW = Math.floor((w - gapPx * (cols - 1)) / cols);
  container.style.gridAutoRows = Math.round(cellW * 1.5) + 'px'; // 2:3 portrait
}

// ---- 6-col bento: only 1×1 (small) and 2×2 (large) squares ----
// 6-card block × 4 = 24 cards, each block = 6 cols × 2 rows = 12 cells, 0 gaps.
// 3 rotating patterns so the large tile walks left→right→center across rows:
//
//   A: [B B][s][s][B B]    B: [B B][B B][s][s]    C: [s][s][B B][B B]
//      [B B][s][s][B B]       [B B][B B][s][s]       [s][s][B B][B B]
//
function applyBentoLayout(cards) {
  const cols   = _bentoColCount();
  const mobile = cols <= 2;

  // On non-6-col breakpoints fall back to uniform flow
  if (mobile || cols === 4) {
    cards.forEach(c => { c.style.gridColumn = ''; c.style.gridRow = ''; });
    return;
  }

  cards.forEach((card, i) => {
    const block  = Math.floor(i / 6);
    const offset = i % 6;
    const rb     = block * 2 + 1; // 1-indexed CSS row start (2 rows per block)
    const pat    = block % 3;     // rotate A(0) → B(1) → C(2) → A …

    // [colStart, colEnd, rowStart, rowEnd] — all 1-indexed CSS grid lines
    const L = {
      0: [[1,3,rb,rb+2],[5,7,rb,rb+2],[3,4,rb,rb+1],[4,5,rb,rb+1],[3,4,rb+1,rb+2],[4,5,rb+1,rb+2]], // A
      1: [[1,3,rb,rb+2],[3,5,rb,rb+2],[5,6,rb,rb+1],[6,7,rb,rb+1],[5,6,rb+1,rb+2],[6,7,rb+1,rb+2]], // B
      2: [[3,5,rb,rb+2],[5,7,rb,rb+2],[1,2,rb,rb+1],[2,3,rb,rb+1],[1,2,rb+1,rb+2],[2,3,rb+1,rb+2]]  // C
    }[pat][offset];

    card.style.gridColumn = `${L[0]}/${L[1]}`;
    card.style.gridRow    = `${L[2]}/${L[3]}`;
  });
}

// Track all active bento grids for resize — Map<container, cards[]>
const _bentoGrids = new Map();

function renderGrid(movies, container = document.getElementById('movies-grid')) {
  if (!container) return;
  container.innerHTML = '';
  if (!movies || movies.length === 0) {
    container.innerHTML = '<p style="color:var(--ink-60);grid-column:1/-1;padding:2rem 0;">No results found.</p>';
    return;
  }
  // Sort highest-rated first → they land in big-tile positions (indices 0 & 1
  // of each 6-card block) so better-poster movies get the large tiles.
  const sorted = [...movies].sort((a, b) => b.rating - a.rating);
  const cards  = sorted.map(m => makeCard(m));
  cards.forEach(card => container.appendChild(card));
  setSquareRows(container);
  applyBentoLayout(cards);
  setupBubbleEffect(container);
  _bentoGrids.set(container, cards);
}

// Keep squares on window resize — update every registered bento grid
window.addEventListener('resize', () => {
  _bentoGrids.forEach((cards, container) => {
    setSquareRows(container);
    applyBentoLayout(cards);
  });
});

function renderRow(container, movies) {
  if (!container) return;
  container.innerHTML = '';
  if (!movies || movies.length === 0) {
    container.innerHTML = '<p style="color:var(--ink-60);padding:1rem 0;">Nothing to show yet.</p>';
    return;
  }
  movies.forEach(m => container.appendChild(makeCard(m)));
}

// ---- Modal Detail ----
async function openDetail(movieId, contentType = 'movie') {
  const overlay = document.getElementById('modal-overlay');
  if (!overlay) { window.location.href = `/movie/${movieId}?type=${contentType}`; return; }
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';

  document.getElementById('modal-title').textContent       = '';
  document.getElementById('modal-meta').innerHTML          = '<div class="spinner" style="margin:0;width:24px;height:24px;"></div>';
  document.getElementById('modal-desc').textContent        = '';
  document.getElementById('modal-actions').innerHTML       = '';
  document.getElementById('modal-director-section').style.display = 'none';
  document.getElementById('modal-cast-section').style.display = 'none';
  document.getElementById('modal-director').textContent    = '';
  document.getElementById('modal-cast').innerHTML          = '';
  document.getElementById('modal-ai').style.display        = 'none';
  document.getElementById('modal-ai-text').textContent     = '';
  document.getElementById('similar-row').innerHTML         = '<div class="spinner" style="margin:0;width:24px;height:24px;"></div>';
  document.getElementById('modal-poster-year').textContent = '';
  document.getElementById('modal-poster-title').textContent = '';
  const posterImg = document.getElementById('modal-poster-img');
  if (posterImg) { posterImg.src = ''; posterImg.style.display = 'none'; }

  try {
    const data = await API.get(`/api/movies/${movieId}?type=${contentType}`);
    const m    = data.movie;

    if (Auth.isLoggedIn()) {
      API.post('/api/recommend/interact', {
        movie_id: movieId, content_type: contentType, action: 'view'
      }).catch(() => {});
    }

    const poster = m.poster || '/static/assets/placeholder.png';

    // Modal poster area
    const modalEl = document.getElementById('modal');
    modalEl.style.setProperty('--mc1', '#2d1810');
    modalEl.style.setProperty('--mc2', '#8b3a1a');

    if (posterImg) {
      posterImg.src = poster;
      posterImg.style.display = '';
      posterImg.onerror = () => { posterImg.style.display = 'none'; };
    }

    const genres = (m.genres || '').split(', ').filter(Boolean);
    document.getElementById('modal-poster-year').textContent = `${m.year || ''} · ${genres.slice(0, 2).join(' · ')}`;
    document.getElementById('modal-poster-title').textContent = m.title;

    // Title with italic after colon
    const titleParts = m.title.split(':');
    document.getElementById('modal-title').innerHTML = titleParts[0] +
      (titleParts.length > 1 ? `<em>: ${titleParts.slice(1).join(':').trim()}</em>` : '');

    const genreChips = genres
      .map(g => `<span class="genre-chip" style="pointer-events:none;">${g}</span>`).join('');
    const tvExtras = contentType === 'tv' ? `
      ${m.seasons  ? `<span class="genre-chip" style="pointer-events:none;">${m.seasons} seasons</span>` : ''}
      ${m.networks ? `<span class="genre-chip" style="pointer-events:none;">${m.networks}</span>` : ''}
      ${m.status   ? `<span class="genre-chip" style="pointer-events:none;">${m.status}</span>` : ''}
    ` : '';

    document.getElementById('modal-meta').innerHTML = `
      <span>★ ${m.rating}</span><span>·</span>
      ${m.year ? `<span>${m.year}</span><span>·</span>` : ''}
      ${contentType === 'tv' ? `<span style="font-size:0.75rem;background:rgba(99,102,241,0.35);border:1px solid rgba(99,102,241,0.5);border-radius:4px;padding:2px 8px;color:#a5b4fc;">TV Show</span>` : ''}
      ${genreChips}${tvExtras}
    `;
    document.getElementById('modal-desc').textContent = m.overview || '';

    // AI explanation (show for all)
    if (m.overview) {
      document.getElementById('modal-ai-text').textContent =
        `Matches your taste profile — high affinity for ${genres.slice(0, 2).join(' and ').toLowerCase() || 'this genre'}.`;
      document.getElementById('modal-ai').style.display = '';
    }

    // Director
    if (m.director) {
      const dirEl = document.getElementById('modal-director');
      dirEl.innerHTML = m.director.split(', ').map(d =>
        `<a href="#" class="director-link" data-name="${d.replace(/"/g, '&quot;')}" style="color:var(--leaf-amber);text-decoration:none;border-bottom:1px dotted var(--leaf-amber);">${d}</a>`
      ).join(', ');
      dirEl.querySelectorAll('.director-link').forEach(link => {
        link.addEventListener('click', e => {
          e.preventDefault();
          const name = link.dataset.name;
          closeDetail();
          const searchInput = document.getElementById('search-input');
          if (searchInput) {
            searchInput.value = name;
            searchInput.dispatchEvent(new Event('input'));
          }
        });
      });
      document.getElementById('modal-director-section').style.display = '';
    }

    // Cast
    if (m.cast_json) {
      try {
        const castList = typeof m.cast_json === 'string' ? JSON.parse(m.cast_json) : m.cast_json;
        if (Array.isArray(castList) && castList.length > 0) {
          const castContainer = document.getElementById('modal-cast');
          castContainer.innerHTML = castList.slice(0, 6).map(c => {
            const name = c.name || '';
            const safeName = name.replace(/"/g, '&quot;');
            const initials = name.split(' ').filter(Boolean).map(w => w[0]).slice(0, 2).join('').toUpperCase();
            const hasPhoto = c.profile && c.profile.length > 5;
            return `
            <div class="cast-card" data-name="${safeName}">
              ${hasPhoto
                ? `<img src="${c.profile}" alt="${safeName}" loading="lazy"
                       onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
                   <div class="cast-avatar" style="display:none;">${initials}</div>`
                : `<div class="cast-avatar">${initials}</div>`
              }
              <div class="cast-name">${name}</div>
              <div class="cast-role">${c.character || ''}</div>
            </div>`;
          }).join('');
          castContainer.querySelectorAll('.cast-card').forEach(card => {
            card.addEventListener('click', () => {
              const name = card.dataset.name;
              if (name) {
                closeDetail();
                const searchInput = document.getElementById('search-input');
                if (searchInput) {
                  searchInput.value = name;
                  searchInput.dispatchEvent(new Event('input'));
                }
              }
            });
          });
          document.getElementById('modal-cast-section').style.display = '';
        }
      } catch (e) {
        if (m.cast) {
          document.getElementById('modal-cast').innerHTML =
            `<p style="font-size:0.85rem;color:var(--ink-60);">${m.cast.replace(/ /g, ', ')}</p>`;
          document.getElementById('modal-cast-section').style.display = '';
        }
      }
    } else if (m.cast) {
      document.getElementById('modal-cast').innerHTML =
        `<p style="font-size:0.85rem;color:var(--ink-60);">${m.cast.replace(/ /g, ', ')}</p>`;
      document.getElementById('modal-cast-section').style.display = '';
    }

    if (Auth.isLoggedIn()) {
      const isFav       = _isFav(m.id, contentType);
      const isWatched   = _isWatched(m.id, contentType);
      const isLiked     = _isLiked(m.id, contentType);
      const isWatchlist = _isWatchlist(m.id, contentType);
      const isNI        = _isNI(m.id, contentType);
      const actions     = document.getElementById('modal-actions');
      actions.innerHTML = `
        <button class="btn-dark" id="dp-like">
          ${isLiked ? '▲ Liked' : '▲ Like'}
        </button>
        <button class="btn-light" id="dp-fav">
          ${isFav ? '♥ Favorited' : '♡ Favorite'}
        </button>
        <button class="btn-light" id="dp-watched">
          ${isWatched ? '✓ Watched' : '○ Watched'}
        </button>
        <button class="btn-light" id="dp-watchlist">
          ${isWatchlist ? '+ In Watchlist' : '+ Watchlist'}
        </button>
        <button class="btn-light" id="dp-ni">
          ${isNI ? '✕ Not Interested' : '✕ Not Interested'}
        </button>
      `;

      document.getElementById('dp-like').addEventListener('click', async () => {
        const btn = document.getElementById('dp-like');
        if (_isLiked(m.id, contentType)) {
          await API.delete(`/api/user/liked/${m.id}?type=${contentType}`).catch(() => {});
          if (contentType === 'tv') userLikedTV     = userLikedTV.filter(x => x !== m.id);
          else                      userLikedMovies = userLikedMovies.filter(x => x !== m.id);
          btn.className = 'btn-light'; btn.textContent = '▲ Like';
          Toast.success('Like removed');
        } else {
          await API.post('/api/user/liked', { movie_id: m.id, content_type: contentType }).catch(() => {});
          if (contentType === 'tv') userLikedTV.push(m.id);
          else                      userLikedMovies.push(m.id);
          btn.className = 'btn-dark'; btn.textContent = '▲ Liked';
          Toast.success('Liked! Your recommendations will update.');
        }
      });

      document.getElementById('dp-fav').addEventListener('click', async () => {
        const btn = document.getElementById('dp-fav');
        if (_isFav(m.id, contentType)) {
          await API.delete(`/api/user/favorites/${m.id}?type=${contentType}`).catch(() => {});
          if (contentType === 'tv') userFavTV     = userFavTV.filter(x => x !== m.id);
          else                      userFavMovies = userFavMovies.filter(x => x !== m.id);
          btn.className = 'btn-light'; btn.textContent = '♡ Favorite';
          Toast.success('Removed from favorites');
        } else {
          await API.post('/api/user/favorites', { movie_id: m.id, content_type: contentType }).catch(() => {});
          if (contentType === 'tv') userFavTV.push(m.id);
          else                      userFavMovies.push(m.id);
          btn.className = 'btn-dark'; btn.textContent = '♥ Favorited';
          Toast.success('Added to favorites');
        }
      });

      document.getElementById('dp-watched').addEventListener('click', async () => {
        const btn = document.getElementById('dp-watched');
        if (_isWatched(m.id, contentType)) {
          await API.delete(`/api/user/watched/${m.id}?type=${contentType}`).catch(() => {});
          if (contentType === 'tv') userWatchedTV     = userWatchedTV.filter(x => x !== m.id);
          else                      userWatchedMovies = userWatchedMovies.filter(x => x !== m.id);
          btn.className = 'btn-light'; btn.textContent = '○ Watched';
          Toast.success('Removed from watched');
        } else {
          await API.post('/api/user/watched', { movie_id: m.id, content_type: contentType }).catch(() => {});
          if (contentType === 'tv') userWatchedTV.push(m.id);
          else                      userWatchedMovies.push(m.id);
          btn.className = 'btn-dark'; btn.textContent = '✓ Watched';
          Toast.success('Marked as watched');
        }
      });

      document.getElementById('dp-watchlist').addEventListener('click', async () => {
        const btn = document.getElementById('dp-watchlist');
        if (_isWatchlist(m.id, contentType)) {
          await API.delete(`/api/user/watchlist/${m.id}?type=${contentType}`).catch(() => {});
          if (contentType === 'tv') userWatchlistTV     = userWatchlistTV.filter(x => x !== m.id);
          else                      userWatchlistMovies = userWatchlistMovies.filter(x => x !== m.id);
          btn.className = 'btn-light'; btn.textContent = '+ Watchlist';
          Toast.success('Removed from watchlist');
        } else {
          await API.post('/api/user/watchlist', { movie_id: m.id, content_type: contentType }).catch(() => {});
          if (contentType === 'tv') userWatchlistTV.push(m.id);
          else                      userWatchlistMovies.push(m.id);
          btn.className = 'btn-dark'; btn.textContent = '+ In Watchlist';
          Toast.success('Added to watchlist');
        }
      });

      document.getElementById('dp-ni').addEventListener('click', async () => {
        const btn = document.getElementById('dp-ni');
        if (_isNI(m.id, contentType)) {
          await API.delete(`/api/user/not-interested/${m.id}?type=${contentType}`).catch(() => {});
          if (contentType === 'tv') userNITV     = userNITV.filter(x => x !== m.id);
          else                      userNIMovies = userNIMovies.filter(x => x !== m.id);
          btn.className = 'btn-light'; btn.textContent = '✕ Not Interested';
          Toast.success('Removed from not interested');
        } else {
          await API.post('/api/user/not-interested', { movie_id: m.id, content_type: contentType }).catch(() => {});
          if (contentType === 'tv') userNITV.push(m.id);
          else                      userNIMovies.push(m.id);
          btn.className = 'btn-dark'; btn.textContent = '✕ Not Interested';
          Toast.success('Not interested — we\'ll show less like this');
        }
      });
    } else {
      document.getElementById('modal-actions').innerHTML =
        `<a href="/auth" class="btn-dark">Sign in to save</a><a href="/movie/${m.id}?type=${contentType}" class="btn-light">View Full</a>`;
    }

    const simData = await API.get(`/api/recommend/similar/${m.id}?type=${contentType}`);
    renderRow(document.getElementById('similar-row'), simData.movies);
  } catch {
    document.getElementById('modal-title').textContent = 'Could not load details';
    document.getElementById('modal-meta').innerHTML    = '';
    document.getElementById('similar-row').innerHTML   = '';
  }
}

function closeDetail() {
  const overlay = document.getElementById('modal-overlay');
  if (overlay) overlay.classList.remove('open');
  document.body.style.overflow = '';
}

document.addEventListener('DOMContentLoaded', () => {
  // Close modal on overlay click
  document.getElementById('modal-overlay')?.addEventListener('click', e => {
    if (e.target.id === 'modal-overlay') closeDetail();
  });
  document.getElementById('modal-close')?.addEventListener('click', closeDetail);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });
  loadUserLists();
});

// ---- List view for profile/watchlist pages ----
function makeListItem(movie) {
  const poster = movie.poster || '/static/assets/placeholder.png';
  const ct     = movie.content_type || 'movie';
  const safeTitle = movie.title.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  const genres = (movie.genres || '').split(', ').filter(Boolean);
  const topGenres = genres.slice(0, 4);
  const maxPct = 100;

  const genreBars = topGenres.map((g, i) => {
    const pct = Math.round(maxPct - i * 18);
    return `<div class="dna-item">
      <div class="dna-label">${g}</div>
      <div class="dna-track"><div class="dna-fill" style="width:${pct}%"></div></div>
      <div class="dna-pct">${pct}%</div>
    </div>`;
  }).join('');

  const tvBadge = ct === 'tv'
    ? `<span style="font-size:0.7rem;background:rgba(99,102,241,0.35);border:1px solid rgba(99,102,241,0.5);border-radius:4px;padding:1px 6px;color:#a5b4fc;">TV</span>`
    : '';

  const el = document.createElement('div');
  el.className = 'list-item';
  el.innerHTML = `
    <div class="list-item-poster" style="cursor:pointer;">
      <img src="${poster}" alt="${safeTitle}" loading="lazy" onerror="this.src='/static/assets/placeholder.png'">
    </div>
    <div class="list-item-info">
      <div class="list-item-header">
        <h3 class="list-item-title" style="cursor:pointer;">${safeTitle}</h3>
        <div class="list-item-meta">
          ${movie.year ? `<span class="genre-chip" style="pointer-events:none;font-size:0.72rem;padding:0.2rem 0.6rem;">${movie.year}</span>` : ''}
          <span class="rating-badge">★ ${movie.rating}</span>
          ${tvBadge}
        </div>
      </div>
      <p class="list-item-overview">${movie.overview || 'No description available.'}</p>
      <div class="list-item-genre-dna">
        <span style="font-size:0.72rem;color:var(--ink-40);margin-bottom:0.3rem;display:block;">Genre DNA</span>
        <div class="genre-dna-grid" style="gap:0.45rem;">
          ${genreBars}
        </div>
      </div>
    </div>
  `;

  el.querySelector('.list-item-poster').addEventListener('click', () => openDetail(movie.id, ct));
  el.querySelector('.list-item-title').addEventListener('click', () => openDetail(movie.id, ct));
  return el;
}

function renderListView(movies, container) {
  if (!container) return;
  container.innerHTML = '';
  if (!movies || movies.length === 0) {
    container.innerHTML = '<p style="color:var(--ink-60);padding:1rem 0;">Nothing here yet.</p>';
    return;
  }
  movies.forEach(m => container.appendChild(makeListItem(m)));
}

// ---- Pagination (called by index.html's loadMovies) ----
function renderPagination(page, pages) {
  const container = document.getElementById('pagination');
  if (!container) return;
  container.innerHTML = '';
  if (pages <= 1) return;

  const mkBtn = (label, p) => {
    const b = document.createElement('button');
    b.className = 'btn btn-ghost';
    b.textContent = label;
    b.addEventListener('click', () => { currentPage = p; loadMovies(); window.scrollTo({ top: 600, behavior: 'smooth' }); });
    return b;
  };

  if (page > 1) container.appendChild(mkBtn('← Prev', page - 1));
  const info = document.createElement('span');
  info.style.cssText = 'color:var(--ink-60);padding:0.5rem 1rem;font-size:0.9rem;';
  info.textContent = `Page ${page} of ${pages}`;
  container.appendChild(info);
  if (page < pages) container.appendChild(mkBtn('Next →', page + 1));
}
