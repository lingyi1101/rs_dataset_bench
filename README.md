The code for evaluating selected remote sensing datasets.
VLMEvalKit_rev contains a revised version of the original codebase. It extends the evaluation support for MME-RealWorld from the original Lite subset to the full benchmark, while retaining native support for XLRS-Bench.

The other directories provide the build, inference, and evaluation pipelines for the datasets indicated by their names. Specifically, they are used for dataset construction and preprocessing, running model inference, and conducting final evaluation, respectively.

这是用于部分遥感数据集的评测代码
VLMEvalKit_rev是修改部分代码后的原版，可以评测mme-realworld（原本只支持lite，我往里加了完整版）和xlrs bench（原生支持）
另外几个包括了标题所写数据集的build infer eval三个部分，分别用于整理数据集，跑推理和最终评测。
