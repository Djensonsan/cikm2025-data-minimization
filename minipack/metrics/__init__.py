# RecPack metrics:
from recpack.metrics.dcg import DCGK, NDCGK
from recpack.metrics.coverage import CoverageK
from recpack.metrics.diversity import IntraListDiversityK
from recpack.metrics.hit import HitK, DiscountedGainK
from recpack.metrics.ips import IPSHitRateK
from recpack.metrics.precision import PrecisionK
from recpack.metrics.recall import RecallK, CalibratedRecallK
from recpack.metrics.reciprocal_rank import ReciprocalRankK
from recpack.metrics.percentile_ranking import PercentileRanking

# New metrics:
from minipack.metrics.ndcg import NDCG

METRICS = {
    "NDCG": NDCG,
    "CoverageK": CoverageK,
    # Note: the RecPack version of NDCG is only correct for binary relevance scores.
    "NDCGK": NDCGK,
    "DCGK": DCGK,
    "IntraListDiversityK": IntraListDiversityK,
    "IPSHitRateK": IPSHitRateK,
    "HitK": HitK,
    "DiscountedGainK": DiscountedGainK,
    "PrecisionK": PrecisionK,
    "RecallK": RecallK,
    "CalibratedRecallK": CalibratedRecallK,
    "ReciprocalRankK": ReciprocalRankK,
    "PercentileRanking": PercentileRanking,
}
