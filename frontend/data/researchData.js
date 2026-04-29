window.researchData = {
  overviewData: {
    purpose: "Explain the research question and end-to-end pipeline.",
    source: "TRAM benchmark + project experiment logs",
    isMock: false,
    lastUpdated: "2026-04-24",
    title: "Research on Complex Temporal Reasoning",
    subtitle: "Classifier-Driven Prompt Routing for TRAM-based Temporal QA",
    intro:
      "This system focuses on temporal reasoning with a small self-trained classifier. Workflow: understand task types, run one-click multi-route tests, then analyze performance boundaries and errors.",
    pipeline: [
      "Input Question",
      "Task Classifier",
      "Confidence Estimation",
      "Prompt Router",
      "Fallback (if low confidence)",
      "LLM Answer",
      "Final Output"
    ]
  },

  threeModelExplorationData: {
    purpose: "Show early baseline heterogeneity as motivation for routing.",
    source: "Early multi-model pilot runs (frontend aligned)",
    isMock: true,
    lastUpdated: "2026-04-24",
    models: [
      { model: "DeepSeek", accuracy: 0.63, parseRate: 0.89, formatCompliance: 0.97, latency: 5.37 },
      { model: "GPT-5-mini", accuracy: 0.61, parseRate: 0.91, formatCompliance: 0.97, latency: 5.82 },
      { model: "Doubao", accuracy: 0.58, parseRate: 0.87, formatCompliance: 0.95, latency: 4.95 }
    ],
    conclusion:
      "Different models behave differently across temporal subcategories. This motivates classifier-based routing and fallback risk control."
  },

  methodComparisonData: {
    purpose: "Compare four deployable workflows under strict split.",
    source: "outputs/tables/prompt_routing_comparison_strict.csv",
    isMock: false,
    lastUpdated: "2026-04-24",
    workflows: [
      { workflow: "Fixed Prompt", accuracy: 0.63, parseRate: 0.8925, formatCompliance: 0.965, latency: 5.3711, callsPerQuery: 1.0 },
      { workflow: "CoT Prompt", accuracy: 0.62, parseRate: 0.9175, formatCompliance: 0.9725, latency: 5.2518, callsPerQuery: 1.0 },
      { workflow: "Classifier Router", accuracy: 0.6075, parseRate: 0.92, formatCompliance: 0.9675, latency: 5.2627, callsPerQuery: 1.0 },
      { workflow: "Classifier Router + Fallback", accuracy: 0.625, parseRate: 0.9075, formatCompliance: 0.9675, latency: 5.3329, callsPerQuery: 1.0 }
    ],
    summary:
      "Router + Fallback is competitive with the strongest baseline. Remaining gap is mainly prompt-bank headroom."
  },

  oracleUpperBoundData: {
    purpose: "Provide non-deployable upper bound for headroom analysis.",
    source: "outputs/tables/third_round_summary.csv",
    isMock: false,
    lastUpdated: "2026-04-24",
    name: "Oracle Prompt Upper Bound",
    accuracy: 0.6325,
    definition: "Upper bound under category-best prompt selection",
    interpretation: "Not a deployable online policy"
  },

  categoryBoundaryData: {
    purpose: "Show where method works, fails, and saturates.",
    source: "outputs/tables/prompt_routing_categorywise_strict.csv",
    isMock: false,
    lastUpdated: "2026-04-24",
    categories: [
      { category: "Date Computation", fixed: 0.5265, cot: 0.5076, router: 0.5038, routerFallback: 0.5265, insightTag: "Recovered by fallback" },
      { category: "Hour Adjustment (24h)", fixed: 0.9545, cot: 0.9773, router: 0.8864, routerFallback: 0.9318, insightTag: "Prompt gap remains" },
      { category: "Time Zone Conversion", fixed: 0.0, cot: 0.0, router: 0.0, routerFallback: 0.0, insightTag: "Shared bottleneck" },
      { category: "Year Shift", fixed: 1.0, cot: 1.0, router: 1.0, routerFallback: 1.0, insightTag: "Ceiling category" },
      { category: "Month Shift", fixed: 0.71, cot: 0.69, router: 0.66, routerFallback: 0.72, insightTag: "Fallback-sensitive" }
    ]
  },

  modelInternalsDemoData: {
    purpose: "Explain how the lightweight TF-IDF + Logistic Regression classifier works step by step.",
    source: "Pedagogical frontend demo aligned with classifier architecture",
    isMock: true,
    lastUpdated: "2026-04-28",
    threshold: 0.95,
    defaultQuestion:
      "If you decrease 3 weeks to the date 10-2-1999, what will be the date in the next year?",
    categories: [
      "Date Computation",
      "Hour Adjustment (24h)",
      "Time Zone Conversion",
      "Year Shift",
      "Month Shift"
    ],
    idf: {
      date: 2.4,
      week: 2.1,
      weeks: 2.1,
      decrease: 2.0,
      next: 1.6,
      year: 1.8,
      "next year": 2.7,
      "3 weeks": 3.0,
      hour: 2.2,
      zone: 3.4,
      utc: 3.5,
      month: 2.4,
      before: 1.9,
      after: 1.9
    },
    biases: {
      "Date Computation": 0.2,
      "Hour Adjustment (24h)": -0.9,
      "Time Zone Conversion": -1.1,
      "Year Shift": -0.2,
      "Month Shift": -0.5
    },
    weights: {
      "Date Computation": {
        date: 0.9,
        week: 0.7,
        weeks: 0.7,
        decrease: 0.35,
        next: 0.22,
        year: 0.15,
        "3 weeks": 1.0,
        "next year": 0.45,
        after: 0.25,
        before: 0.25
      },
      "Hour Adjustment (24h)": {
        hour: 1.2,
        before: 0.1,
        after: 0.1
      },
      "Time Zone Conversion": {
        zone: 1.6,
        utc: 1.8,
        hour: 0.2
      },
      "Year Shift": {
        year: 1.0,
        "next year": 1.15,
        next: 0.45
      },
      "Month Shift": {
        month: 1.3,
        before: 0.35,
        after: 0.35
      }
    }
  },

  analysisRows: [
    {
      sample_id: "row_12130_748f2a27b10e",
      category: "Time Zone Conversion",
      workflow: "Classifier Router + Fallback",
      gold: "01:00",
      pred: "02:00",
      correct: false,
      error_type: "timezone_direction"
    },
    {
      sample_id: "row_12098_e03a432fb227",
      category: "Time Zone Conversion",
      workflow: "Fixed Prompt",
      gold: "18:00",
      pred: "20:00",
      correct: false,
      error_type: "date_rollover"
    },
    {
      sample_id: "row_hour24_demo_01",
      category: "Hour Adjustment (24h)",
      workflow: "Classifier Router",
      gold: "10:38",
      pred: "09:38",
      correct: false,
      error_type: "borrow_error"
    },
    {
      sample_id: "row_date_demo_03",
      category: "Date Computation",
      workflow: "Classifier Router + Fallback",
      gold: "2024-03-01",
      pred: "2024-03-01",
      correct: true,
      error_type: "none"
    }
  ]
};
