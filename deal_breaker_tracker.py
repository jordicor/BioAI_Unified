"""
Deal-Breaker Tracker Module
============================

Tracks Gran Sabio escalations and maintains reliability statistics for QA models.
Provides alerts when models show excessive false positive rates.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
# Use optimized JSON
import json_utils as json

from models import GranSabioEscalation, ModelReliabilityStats
from config import config

logger = logging.getLogger(__name__)


class DealBreakerTracker:
    """
    Centralized tracking for deal-breaker escalations and model reliability.
    Maintains in-memory state and persists to disk.
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path or config.TRACKING_DATA_PATH
        Path(self.persistence_path).mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self.escalations: Dict[str, List[GranSabioEscalation]] = {}  # session_id -> [escalations]
        self.model_stats: Dict[str, ModelReliabilityStats] = {}  # model_name -> stats

        # Load from disk if exists
        self._load_model_stats()

    def record_escalation(
        self,
        session_id: str,
        iteration: int,
        layer_name: str,
        trigger_type: str,
        triggering_model: str,
        deal_breaker_reason: str,
        total_models: int,
        deal_breaker_count: int,
        gran_sabio_model: str
    ) -> str:
        """
        Record a new Gran Sabio escalation

        Returns:
            escalation_id: Unique ID for this escalation
        """
        escalation_id = f"{session_id}_escalation_{len(self.escalations.get(session_id, []))}"

        escalation = GranSabioEscalation(
            escalation_id=escalation_id,
            session_id=session_id,
            iteration=iteration,
            layer_name=layer_name,
            trigger_type=trigger_type,
            triggering_model=triggering_model,
            deal_breaker_reason=deal_breaker_reason,
            total_models_evaluated=total_models,
            deal_breaker_count=deal_breaker_count,
            gran_sabio_model_used=gran_sabio_model
        )

        if session_id not in self.escalations:
            self.escalations[session_id] = []

        self.escalations[session_id].append(escalation)

        logger.info(f"Recorded escalation {escalation_id} for session {session_id}")

        return escalation_id

    def complete_escalation(
        self,
        escalation_id: str,
        decision: str,
        reasoning: str,
        was_real: Optional[bool]
    ):
        """
        Mark escalation as complete and update model statistics
        """
        # Find the escalation
        escalation = None
        for session_escalations in self.escalations.values():
            for esc in session_escalations:
                if esc.escalation_id == escalation_id:
                    escalation = esc
                    break
            if escalation:
                break

        if not escalation:
            logger.error(f"Escalation {escalation_id} not found")
            return

        # Update escalation
        escalation.decision = decision
        escalation.reasoning = reasoning
        escalation.was_real_deal_breaker = was_real
        escalation.completed_at = datetime.now()
        escalation.duration_seconds = (
            escalation.completed_at - escalation.started_at
        ).total_seconds()

        # Update model statistics
        model_name = escalation.triggering_model
        self._update_model_stats(model_name, was_real)

        logger.info(
            f"Completed escalation {escalation_id}: "
            f"decision={decision}, was_real={was_real}"
        )

    def _update_model_stats(self, model_name: str, was_real_deal_breaker: Optional[bool]):
        """Update statistics for a model"""
        if was_real_deal_breaker is None:
            logger.info(
                "Skipping reliability stats update for model %s due to unresolved Gran Sabio decision.",
                model_name
            )
            return
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelReliabilityStats(
                qa_model_name=model_name
            )

        stats = self.model_stats[model_name]
        stats.total_deal_breakers_raised += 1

        if was_real_deal_breaker:
            stats.confirmed_real += 1
        else:
            stats.confirmed_false_positive += 1

        # Recalculate metrics
        stats.calculate_metrics()

        # Check for alert threshold
        if stats.confirmed_real + stats.confirmed_false_positive >= config.MODEL_RELIABILITY_MIN_SAMPLES:
            if stats.false_positive_rate >= config.MODEL_RELIABILITY_FALSE_POSITIVE_THRESHOLD:
                self._trigger_alert(model_name, stats)

        # Persist to disk
        self._save_model_stats()

    def _trigger_alert(self, model_name: str, stats: ModelReliabilityStats):
        """
        Trigger alert when model exceeds false positive threshold
        """
        logger.warning(
            f"ALERT: Model {model_name} has high false positive rate: "
            f"{stats.false_positive_rate:.2%} "
            f"({stats.confirmed_false_positive}/{stats.total_deal_breakers_raised} false positives). "
            f"Reliability badge: {stats.reliability_badge}. "
            f"Consider reviewing this model's configuration."
        )

    def get_session_escalations(self, session_id: str) -> List[GranSabioEscalation]:
        """Get all escalations for a session"""
        return self.escalations.get(session_id, [])

    def get_session_escalation_count(self, session_id: str) -> int:
        """Get total escalation count for a session"""
        return len(self.escalations.get(session_id, []))

    def get_model_stats(self, model_name: str) -> Optional[ModelReliabilityStats]:
        """Get reliability stats for a specific model"""
        return self.model_stats.get(model_name)

    def get_all_model_stats(self) -> Dict[str, ModelReliabilityStats]:
        """Get all model reliability statistics"""
        return self.model_stats

    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics summary

        Returns:
            Dict with overall statistics and model breakdown
        """
        total_escalations = sum(len(esc) for esc in self.escalations.values())
        total_real = sum(
            sum(1 for e in esc_list if e.was_real_deal_breaker is True)
            for esc_list in self.escalations.values()
        )
        total_false_positives = sum(
            sum(1 for e in esc_list if e.was_real_deal_breaker is False)
            for esc_list in self.escalations.values()
        )
        total_unresolved = sum(
            sum(1 for e in esc_list if e.was_real_deal_breaker is None)
            for esc_list in self.escalations.values()
        )
        resolved_escalations = total_real + total_false_positives

        return {
            "total_escalations": total_escalations,
            "total_real_deal_breakers": total_real,
            "total_false_positives": total_false_positives,
            "total_unresolved": total_unresolved,
            "overall_false_positive_rate": (
                total_false_positives / resolved_escalations if resolved_escalations > 0 else 0.0
            ),
            "model_statistics": {
                name: {
                    "total_raised": stats.total_deal_breakers_raised,
                    "confirmed_real": stats.confirmed_real,
                    "confirmed_false_positive": stats.confirmed_false_positive,
                    "false_positive_rate": stats.false_positive_rate,
                    "precision": stats.precision,
                    "reliability_badge": stats.reliability_badge,
                    "last_updated": stats.last_updated.isoformat()
                }
                for name, stats in self.model_stats.items()
            },
            "high_reliability_models": [
                name for name, stats in self.model_stats.items()
                if stats.reliability_badge == "HIGH"
            ],
            "low_reliability_models": [
                name for name, stats in self.model_stats.items()
                if stats.reliability_badge == "LOW"
            ]
        }

    def _save_model_stats(self):
        """Persist model statistics to disk"""
        stats_file = os.path.join(self.persistence_path, "model_reliability_stats.json")

        try:
            data = {
                name: stats.model_dump(by_alias=True)
                for name, stats in self.model_stats.items()
            }

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved model stats to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save model stats: {e}")

    def _load_model_stats(self):
        """Load model statistics from disk"""
        stats_file = os.path.join(self.persistence_path, "model_reliability_stats.json")

        if not os.path.exists(stats_file):
            logger.info("No existing model stats file found, starting fresh")
            return

        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for name, stats_dict in data.items():
                self.model_stats[name] = ModelReliabilityStats(**stats_dict)

            logger.info(f"Loaded stats for {len(self.model_stats)} models from {stats_file}")
        except Exception as e:
            logger.error(f"Failed to load model stats: {e}")


# Global tracker instance
_tracker_instance: Optional[DealBreakerTracker] = None


def get_tracker() -> DealBreakerTracker:
    """Get or create the global tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = DealBreakerTracker()
    return _tracker_instance
