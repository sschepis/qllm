// Math utility functions for the Resonant Knowledge Model

/**
 * Calculate digital root (modulo 9)
 * @param {number} n - Number to calculate digital root for
 * @returns {number} - Digital root (1-9)
 */
function digitalRoot(n) {
  if (n === 0) return 0;
  return 1 + ((n - 1) % 9);
}

module.exports = {
  digitalRoot
};