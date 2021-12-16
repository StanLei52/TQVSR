import torch
import torch.nn as nn
import sys
sys.path.append(".")
from AQVSR.pytorch_bert.VisualBert import VisualBertConfig
from AQVSR.pytorch_bert.Bert import BertConfig

CTX_ADDING_FUSION_CFG = VisualBertConfig(
    num_hidden_layers = 4,
    visual_embedding_dim = 2048,
    embedding_type = "adding_fusion",
    use_cls_token = True, 
    use_token_type = True
)

QUERY_ADDING_FUSION_CFG = VisualBertConfig(
    num_hidden_layers = 4,
    visual_embedding_dim = 2048,
    embedding_type = "adding_fusion",
    use_cls_token = False,
    use_token_type = True
)

############################################################
CTX_CONCAT_FUSION_CFG = VisualBertConfig(
    num_hidden_layers=2,
    visual_embedding_dim = 2048,
    embedding_type = "vt_concat",
    use_cls_token = True,
    use_token_type = True
)

CTX_CONCAT_NO_CLS_CFG = VisualBertConfig(
    num_hidden_layers=2,
    visual_embedding_dim=2048,
    embedding_type="vt_concat",
    use_cls_token=False,
    use_token_type=True
)

QUERY_CONCAT_FUSION_CFG = VisualBertConfig(
    num_hidden_layers=2,
    visual_embedding_dim = 2048,
    embedding_type = "query_concat",
    use_cls_token = False,
    use_token_type = True
)

MM_FUSION_CFG = BertConfig(
    num_hidden_layers=2,
)



