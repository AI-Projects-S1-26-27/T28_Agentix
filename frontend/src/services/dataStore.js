// Shared in-memory store — persists across page navigation (module singleton)
// Pages read initial state from here so unmount/remount doesn't reset charts

const store = {
  latest: null,
  riskData: null,
  riskHistory: [],    // shared by AIRiskAnalysis
  history: [],        // shared by Dashboard + HealthAnalytics
  envHistory: [],     // shared by EnvironmentalIntelligence (same data, separate slice)
}

export { store }
export default store
