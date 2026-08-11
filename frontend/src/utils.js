// ─────────────────────────────────────────────────────────────
// Neural Studio X — Utilities (src/utils.js)
// Reproducible synthetic dataset generation matching ml_utils.py
// ─────────────────────────────────────────────────────────────

export function getHouseData(n = 800) {
  const data = [];
  // Simple LCG pseudo-random generator to ensure same seed=42 results
  let seed = 42;
  const random = () => {
    const x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
  };

  const neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'];

  for (let i = 0; i < n; i++) {
    const grLivArea = Math.floor(600 + random() * 3400);
    const overallQual = Math.floor(1 + random() * 10);
    const totalBsmtSF = Math.floor(random() * 2500);
    const yearBuilt = Math.floor(1940 + random() * 83);
    const fullBath = Math.floor(1 + random() * 3);
    const halfBath = Math.floor(random() * 2);
    const neighborhood = neighborhoods[Math.floor(random() * neighborhoods.length)];

    // Pricing math matching Python exactly
    const basePrice = 30000 + grLivArea * 65 + overallQual * 16000 + totalBsmtSF * 45 + (yearBuilt - 1940) * 560 + fullBath * 7500;
    const noise = (random() - 0.5) * 22000; // normal approximation
    const price = Math.max(basePrice + noise, 50000);

    data.push({
      id: 1461 + i,
      GrLivArea: grLivArea,
      OverallQual: overallQual,
      TotalBsmtSF: totalBsmtSF,
      YearBuilt: yearBuilt,
      FullBath: fullBath,
      HalfBath: halfBath,
      Neighborhood: neighborhood,
      SalePrice: Math.round(price)
    });
  }
  return data;
}

export function buildFeatures(row) {
  const TotalSF = (row.TotalBsmtSF || 0) + (row.GrLivArea || 0);
  const TotalBath = (row.FullBath || 0) + 0.5 * (row.HalfBath || 0);
  const HouseAge = 2026 - (row.YearBuilt || 2000);
  return { ...row, TotalSF, TotalBath, HouseAge };
}
