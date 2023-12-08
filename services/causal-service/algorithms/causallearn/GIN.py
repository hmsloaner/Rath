from typing import Dict, List, Tuple, Optional, Union, Literal
import traceback
from pydantic import Field

from algorithms.common import getOpts, IDepTestItems, ICatEncodeType, IQuantEncodeType, UCPriorityItems, UCRuleItems, AlgoInterface, IRow, IDataSource, IFieldMeta, IFields
from algorithms.common import transDataSource, OptionalParams
import algorithms.common as common
import numpy as np

from causallearn.search.HiddenCausal.GIN.GIN import GIN as gin
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

class GINParams(OptionalParams, title="GIN Algorithm(暂不支持背景知识)"):
    """
    G: GeneralGraph. Causal graph.
    K: list. Causal Order.
    """
    indep_test_method: Optional[str] = Field(
        default='kci', title="独立性检验", #"Independence Test",
        description="The independence test to use for causal discovery",
        options=getOpts({'kci': 'kci', 'hsic': 'hsic'})
    )
    alpha: Optional[float] = Field(
        default=0.05, title="显著性阈值", # "Alpha",
        description="desired significance level (float) in (0, 1). Default: 0.05.",
        gt=0.0, le=1.0
    )

class GIN(AlgoInterface):
    ParamType = GINParams
    def __init__(self, dataSource: List[IRow], fields: List[IFieldMeta], params: Optional[ParamType] = ParamType()):
        print("GIN", dataSource, fields, params)
        super(GIN, self).__init__(dataSource=dataSource, fields=fields, params=params)
        
    def constructBgKnowledge(self, bgKnowledges: Optional[List[common.BgKnowledge]] = None, f_ind: Optional[Dict[str, int]] = None):
        bgKnowledges = [] if bgKnowledges is None else bgKnowledges
        f_ind = {} if f_ind is None else f_ind
        node = self.cg.G.get_nodes()
        # self.bk = BackgroundKnowledge()
        # for k in bgKnowledges:
        #     if k.type > common.bgKnowledge_threshold[1]:
        #         self.bk.add_required_by_node(node[f_ind[k.src]], node[f_ind[k.tar]])
        #     elif k.type < common.bgKnowledge_threshold[0]:
        #         self.bk.add_forbidden_by_node(node[f_ind[k.src]], node[f_ind[k.tar]])
        # return self.bk
        
    def calc(self, params: Optional[ParamType] = ParamType(), focusedFields: Optional[List[str]] = None, bgKnowledgesPag: Optional[List[common.BgKnowledgePag]] = None, **kwargs):
        focusedFields = [] if focusedFields is None else focusedFields
        bgKnowledgesPag = [] if bgKnowledgesPag is None else bgKnowledgesPag
        array = self.selectArray(focusedFields=focusedFields, params=params)
        # common.checkLinearCorr(array)
        params.__dict__['cache_path'] = None # '/tmp/causal/pc.json'
        G, K = gin(array, params.indep_test_method, params.alpha)
    
        return {
            'data': G.graph.tolist(),
            'matrix': G.graph.tolist(),
            'fields': self.safeFieldMeta(self.focusedFields),
            'causalOrder': K
        }
