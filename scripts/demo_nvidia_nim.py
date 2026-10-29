"""
Interactive Demonstration & Verification Script for NVIDIA NIM BGP Advisor.
Tests 4 real-world BGP scenarios through NVIDIA NIM:
1. Normal Route Propagation
2. Benign Route Flapping (Multi-homing churn)
3. Gao-Rexford Route Leak (Valley-free violation)
4. Malicious Prefix Hijack (Rogue origin AS divergence)
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ai.nvidia_nim import NvidiaNimBGPAdvisor

def run_demo():
    print("=" * 80)
    print("AI-ENHANCED BGP AUTONOMOUS CONTROL PLANE — NVIDIA NIM ADVISOR EVALUATION")
    print("=" * 80)

    advisor = NvidiaNimBGPAdvisor()
    print(f"[*] Base URL : {advisor.base_url}")
    print(f"[*] Model    : {advisor.model}")
    print(f"[*] API Key  : {'Configured (' + advisor.api_key[:10] + '...)' if advisor.is_configured() else 'Missing'}")
    print("-" * 80)

    test_cases = [
        {
            "name": "Scenario 1: Normal Legitimate Route",
            "prefix": "203.0.113.0/24",
            "as_path": [65001, 65003, 65009],
            "origin_as": 65009,
            "expected_origin_as": 65009,
            "valley_free_violation": False,
            "announcements_per_minute": 0.2,
            "flap_count_5min": 0,
            "loc_pref_current": 100,
            "ml_prediction": "Normal",
            "ml_probabilities": [0.94, 0.04, 0.01, 0.01]
        },
        {
            "name": "Scenario 2: Benign Multi-Homing Route Flap (Transient Churn)",
            "prefix": "198.51.100.0/24",
            "as_path": [65001, 65004, 65008],
            "origin_as": 65008,
            "expected_origin_as": 65008,
            "valley_free_violation": False,
            "announcements_per_minute": 35.0,
            "flap_count_5min": 14,
            "loc_pref_current": 100,
            "ml_prediction": "Suspicious",
            "ml_probabilities": [0.20, 0.72, 0.05, 0.03]
        },
        {
            "name": "Scenario 3: Valley-Free Route Leak (Transit Commercial Leak)",
            "prefix": "192.0.2.0/24",
            "as_path": [65001, 65002, 65005, 65006, 65007],
            "origin_as": 65007,
            "expected_origin_as": 65007,
            "valley_free_violation": True,
            "announcements_per_minute": 1.5,
            "flap_count_5min": 1,
            "loc_pref_current": 100,
            "ml_prediction": "Route Leak",
            "ml_probabilities": [0.08, 0.12, 0.78, 0.02]
        },
        {
            "name": "Scenario 4: Rogue Origin Prefix Hijack (Rogue AS65010)",
            "prefix": "192.0.2.0/24",
            "as_path": [65001, 65004, 65010],
            "origin_as": 65010,
            "expected_origin_as": 65007,
            "valley_free_violation": False,
            "announcements_per_minute": 60.0,
            "flap_count_5min": 8,
            "loc_pref_current": 100,
            "ml_prediction": "Prefix Hijack",
            "ml_probabilities": [0.01, 0.04, 0.05, 0.90]
        }
    ]

    for tc in test_cases:
        print(f"\n>>> Running: {tc['name']}")
        print(f"    Target Prefix   : {tc['prefix']}")
        print(f"    Announced Path  : {tc['as_path']} (Origin: AS{tc['origin_as']})")
        print(f"    Expected Origin : AS{tc['expected_origin_as']}")
        print(f"    Valley-Free Err : {tc['valley_free_violation']}")
        print(f"    Churn Rate      : {tc['announcements_per_minute']} updates/min")

        t0 = time.time()
        result = advisor.analyze_route(
            prefix=tc["prefix"],
            as_path=tc["as_path"],
            origin_as=tc["origin_as"],
            expected_origin_as=tc["expected_origin_as"],
            valley_free_violation=tc["valley_free_violation"],
            announcements_per_minute=tc["announcements_per_minute"],
            flap_count_5min=tc["flap_count_5min"],
            loc_pref_current=tc["loc_pref_current"],
            ml_prediction=tc["ml_prediction"],
            ml_probabilities=tc["ml_probabilities"]
        )
        elapsed = (time.time() - t0) * 1000.0

        print(f"    [NVIDIA NIM Result] ({elapsed:.1f} ms) [Source: {result.get('source')}]:")
        print(f"      - Classification     : {result.get('classification')}")
        print(f"      - Confidence         : {result.get('confidence')}")
        print(f"      - Trust Score        : {result.get('trust_score')}")
        print(f"      - Recommended Policy : {result.get('recommended_policy')} (LocalPref -> {result.get('suggested_local_pref')})")
        print(f"      - RFC Grounding      : {result.get('rfc_reference')}")
        print(f"      - Technical Reason   : {result.get('explanation')}")

    print("\n" + "=" * 80)
    print("NVIDIA NIM ADVISOR VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
