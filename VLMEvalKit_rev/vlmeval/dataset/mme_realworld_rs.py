# vlmeval/dataset/mme_realworld_rs.py
from .image_mcq import MMERealWorld

class MMERealWorld_RS(MMERealWorld):
    """
    Filtered subset of MME-RealWorld: only Remote Sensing items.
    """

    # 关键：给新 dataset 名补上 SYS / MD5 的映射（复用原始 MME-RealWorld）
    SYS = dict(MMERealWorld.SYS)
    SYS["MME-RealWorld-RS"] = MMERealWorld.SYS["MME-RealWorld"]

    DATASET_MD5 = dict(MMERealWorld.DATASET_MD5)
    DATASET_MD5["MME-RealWorld-RS"] = MMERealWorld.DATASET_MD5["MME-RealWorld"]

    @classmethod
    def supported_datasets(cls):
        return ["MME-RealWorld-RS"]

    def load_data(self, dataset="MME-RealWorld-RS", repo_id="yifanzhang114/MME-RealWorld-Base64"):
        base_df = super().load_data(dataset="MME-RealWorld", repo_id=repo_id)
        rs_df = base_df[base_df["category"] == "Perception/Remote Sensing"].copy()

        # 关键：让 dataset_name 真的是新名字，这样 outputs 文件名也会带 RS
        self.dataset_name = "MME-RealWorld-RS"
        return rs_df
