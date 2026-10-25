/**
 * Signora Service Worker
 * Caches static assets for offline support.
 */

const CACHE_NAME = 'signora-v22';
const STATIC_ASSETS = [
    '/',
    '/text-to-sign',
    '/static/css/style.css',
    '/static/css/signora.css',
    '/static/css/text-to-sign.css',
    '/static/img/signora-logo.svg',
    '/static/img/bg-signs.webp',
    '/static/js/app.js?v=14',
    '/static/js/text-to-sign.js?v=7',
    '/static/js/emergency-alert.js?v=3',
    '/static/js/camera.js?v=3',
    '/static/manifest.json',
    'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js'
];

// Pre-cache gesture reference images (A-Z, 0-9, words)
const GESTURE_IMAGES = [];
'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('').forEach(c => {
    GESTURE_IMAGES.push(`/static/gestures/${c}.png`);
});
// Word gesture images
['HELLO', 'YES', 'NO', 'LOVE', 'SPACE', 'GOODBYE', 'THANKYOU', 'PLEASE', 'SORRY',
    'HAPPY', 'SAD', 'MOTHER', 'FATHER', 'BROTHER', 'SISTER', 'FRIEND', 'FAMILY',
    'TEACHER', 'FOOD', 'WATER', 'HOUSE', 'MONEY', 'TIME', 'MORNING', 'NIGHT',
    'DOCTOR', 'COLLEGE', 'AFRAID', 'AGREE', 'ASSISTANCE', 'BAD', 'BECOME', 'FROM',
    'PAIN', 'PRAY', 'SECONDARY', 'SKIN', 'SMALL', 'SPECIFIC', 'STAND', 'TODAY',
    'WARN', 'WHICH', 'WORK', 'YOU'].forEach(w => {
        GESTURE_IMAGES.push(`/static/gestures/${w}.png`);
    });

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll([...STATIC_ASSETS, ...GESTURE_IMAGES]);
        }).catch(err => {
            console.warn('SW: Some assets failed to cache', err);
            // Still install even if some assets fail
            return caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS));
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Don't cache API calls, video feed, sign videos, or POST requests
    if (event.request.method !== 'GET' ||
        url.pathname.startsWith('/api/') ||
        url.pathname === '/video_feed' ||
        url.pathname.startsWith('/text-to-sign/videos/')) {
        return;
    }

    event.respondWith(
        caches.match(event.request).then(cached => {
            if (cached) return cached;
            return fetch(event.request).then(response => {
                // Cache successful GET responses for static assets
                if (response.ok && (
                    url.pathname.startsWith('/static/') ||
                    url.hostname.includes('cdnjs.cloudflare.com')
                )) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                }
                return response;
            });
        }).catch(() => {
            // Offline fallback — serve cached index
            if (event.request.mode === 'navigate') {
                return caches.match('/');
            }
        })
    );
});
