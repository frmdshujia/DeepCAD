/**
 * Deep CAD Landing Page - Hero scroll & entrance + intro image float
 */

const heroTitle = document.getElementById('heroTitle');
const heroSubtitle = document.querySelector('.hero-subtitle');
const introImg = document.querySelector('.intro-scheme-img');

function updateHeroScale() {
  if (!heroTitle) return;
  const scrollY = window.scrollY;
  const progress = Math.min(scrollY / 500, 1);
  const scale = 0.82 + 0.4 * progress;
  const opacity = 0.72 + 0.28 * progress;
  heroTitle.style.transform = `scale(${scale})`;
  heroTitle.style.opacity = opacity;
  if (heroSubtitle) heroSubtitle.style.opacity = opacity;

  // Intro image: float up and grow as user scrolls (scrollY 300–900)
  if (introImg) {
    const imgProgress = Math.min(Math.max((scrollY - 300) / 550, 0), 1);
    const imgScale = 0.78 + 0.22 * imgProgress;
    const imgOpacity = 0.45 + 0.55 * imgProgress;
    const imgY = 60 - 60 * imgProgress;
    introImg.style.transform = `translateY(${imgY}px) scale(${imgScale})`;
    introImg.style.opacity = imgOpacity;
  }
}

if (heroTitle) {
  heroTitle.style.opacity = '0';
  heroTitle.style.transform = 'scale(0.75)';
  requestAnimationFrame(() => {
    heroTitle.style.transition = 'opacity 0.9s cubic-bezier(0.22, 1, 0.36, 1), transform 0.9s cubic-bezier(0.22, 1, 0.36, 1)';
    heroTitle.style.opacity = '0.72';
    heroTitle.style.transform = 'scale(0.82)';
  });
  setTimeout(() => {
    heroTitle.style.transition = 'transform 0.12s ease-out, opacity 0.12s ease-out';
  }, 1000);
}

window.addEventListener('scroll', updateHeroScale, { passive: true });
window.addEventListener('load', updateHeroScale);
