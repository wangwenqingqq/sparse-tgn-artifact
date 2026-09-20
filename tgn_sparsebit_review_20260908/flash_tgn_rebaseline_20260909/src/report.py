from pathlib import Path
import json
r=Path(__file__).resolve().parents[1]
s=json.loads((r/'analysis/summary.json').read_text());k=json.loads((r/'analysis/kernel_findings.json').read_text())
text='''**本次已按 FlashTGN 重做。主要结论是：旧缓存路线应停止作为主线；真正值得继续调查的是 W_kv 投影的反向计算。但当前这份 FlashTGN artifact 尚未通过预先固定的数值验证门槛，不能宣称任何相对 FlashTGN 的优化已经合格。**

这次实际执行的是已有 FlashTGN 的 `train_flash_tgn.py → FlashTGNTrainer.train/run_epoch`，不是此前的 TNCN。此前的 1.096× 等数字仍只代表 TNCN 内部消融，不是相对 FlashTGN 的收益。原 FlashTGN 源码、已编译扩展、数据及此前封存实验均未修改；下文修复只装在独立诊断对象上。

使用完整 Wikipedia 输入：157,474 条时间有序交互、9,227 个节点、原始 172 维边特征、256 维零节点特征。CPU 检查了数据哈希、TCSR 的端点/事件编号/时间和每条交互的两个方向。固定 batch_size=200/2000、邻居数=20/50、层数=1/2，共八种配置；embedding/time 维度 256、双头、dropout=0.1、Adam lr=1e-4、seed=42。保持源码 FP32 + matmul precision='high' 的精度策略，未套用旧 TNCN 的 TF32-off 设置。源码 auto 策略在单层使用 GPU schedule 预计算、双层使用在线 schedule。

源副本没有 Git 元数据，因此不把历史提交标签当成已核实的上游版本。源码、实际扩展二进制、数据和隔离依赖共 809 个文件由 [input_manifest.json](input_manifest.json) 固定。环境为 Python 3.12.3、PyTorch 2.11.0+cu130、实际 scikit-learn 1.7.2；没有使用旧工程里会返回伪指标的训练专用 shim。原版资格检查在 GPU3 上进行；随后其他任务占用 GPU3，按事前记录的修订切到同型号 GPU0。所有计时、阶段剖析和内核追踪统一使用 GPU0，RTX PRO 6000 Blackwell Server Edition。其他 GPU 的任务和共享服务未动；主机 CPU/内存并非专用隔离环境。

**首先发现了实际 mailbox 写入问题。**

每个配置先训练 8 步，从同一个完整 checkpoint 恢复，分别执行原版 A、原版 B、带事件计时的原版各 4 个连续步骤。逐步归档参数、梯度、Adam、memory/mailbox、预测、loss、embedding 和 RNG；单层 124 项、双层 174 项。浮点门槛事前固定为 `abs(a-b) <= 2e-4 + 2e-4*abs(reference)`，离散量和 RNG 必须相同，且所有浮点值有限。原版自重放和计时透明性合计 **0/64** 比较通过；独立 CPU/NumPy 复核确认了每一个失败字段。同时记录的真实 temporal sampler 返回值，3,584 个查询行、56 次调用全部通过 CPU 重建检查。该采样审计是抽查，不是所有内核的完整正确性证明。

首步的预测、loss、embedding、mem_data、mem_ts 和 RNG 在原版 A/B 间逐字节相同。大幅差异出现在随后的 `mail_data[nodes] = mails` 与 `mail_ts[nodes] = times`；其中两个配置还出现小幅反向梯度越界。这使“恢复漏了状态”不符合首处分歧的证据。

源码 [state.py](sources/flash_tgn/state.py) 声明 last-write-wins，却直接使用含重复节点的 CUDA 索引赋值。PyTorch 明确说明非累加索引赋值遇到重复索引时行为未定义；这种写法不保证“最后一条完整消息生效”。[PyTorch 2.11 index_put_ 文档](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.index_put_.html)

独立 CPU 按源码实际 `[src,dst]` 拼接顺序逐条赋值，并用归档的共同 memory 和原始边特征重建 payload。batch_size=200 的首步触达 126 个节点，其中 68 个节点重复；batch_size=2000 触达 769 个节点，其中 483 个重复。原版首步归档中，分别有 44–48、303–318 行 payload 甚至不等于该节点的任何一条完整输入消息，表明出现了消息内容混写，不能解释为正常的浮点舍入。

随后对八组完全固定的输入各重复 16 次写入：

|写入方式|符合 CPU 顺序语义|每组 16 次的不同输出数|
|---|---:|---:|
|原版重复索引赋值|0/128|16|
|仅 writer 内启用 PyTorch deterministic|128/128|1|
|显式选取每个节点最后位置，再唯一索引赋值|128/128|1|

完整原始输出保留，独立 CPU 审计与上述统计全部一致。“最后位置”按原代码拼接顺序定义；没有偷换成任意时间戳最大值规则。

**修复 mailbox 还不等于建立合格基线。**

独立修复只把上述两次赋值替换为整数 `amax` 选最后位置、再唯一索引写入，payload 计算及其他训练操作保持源码。修复后的首步 forward、memory 和 mailbox 在自重放中恢复了字节一致，但首步 backward 仍可有小幅差异，后续时间编码梯度和训练轨迹进一步分离。相同门槛下合计 **11/64** 比较通过，八种配置都没有完整通过四步重放。没有放宽容差、筛选随机种子或将过程退出成功写成数值成功。

已执行的融合反向源码含浮点 atomicAdd；这与反向非确定性相符。不过本次尚未逐个冻结内核输入来证明全部剩余误差的唯一来源，也没有证明训练质量变差。应分别处理明确的 mailbox 竞争写入和反向数值噪声；不能把两者混成一个问题，也不能用训练 loss 有限来替代验证。

**以下性能数据仅作原版 FlashTGN 诊断。**

资格失败后，[AMENDMENT_1.md](AMENDMENT_1.md) 明确只允许诊断计时，原来的性能资格 gate 保持关闭。实际计时运行的仍是未修复的原版。因此下表能回答这份实现把时间花在哪里，不能用来证明候选准确性、训练质量一致或优化收益。

三次独立进程、每种配置 4 个完整训练 epoch，共 96 个 epoch、29,088 个 batch。每次保留第 1 个 epoch 为冷启动/预热证据，汇总其余 9 个观测/配置。训练划分依源码按 batch 对齐：batch_size=200 为 110,200 条交互/551 步，batch_size=2000 为 110,000 条/55 步，未使用 debug 截断。下表不是对不同 batch_size 的质量等价速度排名。

整 epoch 计时覆盖 reset、schedule 刷新与释放、源码 empty_cache、在线训练及源码每轮日志。源码自己打印的 loop 时间不含外部 schedule 准备。构造 Trainer 与 train 调用前的初始 schedule 单列保留；train 总调用墙钟也另行记录，没有隐去。

'''+(r/'analysis/timing_table.md').read_text()+'''
表中 schedule 列仅指单层在在线循环之外的预计算；双层该列为零，表示采样/构造包含在 online 内，并非免费。p10/p90 是 9 个观测的描述性分位数，不是跨机器或所有训练轨迹的置信保证。原始每轮耗时、初始化、首次 schedule、loss、分配/保留显存及调用计数见 [summary.json](analysis/summary.json) 与 [timing_epochs.json](analysis/timing_epochs.json)。32 份完整训练末状态另经 CPU 审计均有限；AP/AUC 未评估，train-only 输出的零不是质量结果。

另做 16 个完整 epoch、4,848 个 batch 的事件剖析，以下使用每个配置的第 2 个 epoch。比例以同一次带计时的完整 epoch 为分母，父子 scope 不重复相加。事件包含主机提交间隙和插桩影响，不能当成纯 GPU kernel 占比。

'''+(r/'analysis/profile_table.md').read_text()+'''
mailbox 写入只有 **0.4%–2.2%**。按此剖面假设它完全免费，整轮也仅约 **1.004–1.022×**，不支持把它当成整体 10% 收益的主方向。这是剖面上的理想化估算，不是硬件严格上界或已经实现的加速。

FlashTGN 已经使用固定 GPU memory/mailbox 表和实际执行的 `fused_lse_gather`，没有旧 TNCN 那条 Python 字典逐节点拆包/拼接读取路径。更关键的是，FlashTGN mailbox 保存的是发送时 memory 与边特征拼接后的值；不能仅用原始 event ID 在以后重新读取当前 memory 来无损重建。此前的 event-index 方案不能直接移植。

memory update 整段约 1.4%–9.8%，还包含真实时间编码、GRU 与状态写回，并不都是可省掉的搬运。backward 占 **39.1%–52.7%**，embedding 占 **14.7%–37.5%**，是更有根据的后续调查方向。假设整个 backward 耗时减半，剖面投影约 1.243–1.358×；这只说明容量值得调查，并未证明实现得出这种收益。

实际扩展监控和源码路径计数确认 fused gather、LSE、dense/segmented attention 前后向都执行过；双层还执行 L0 gather 与 inverse scatter。资格、计时和剖析均没有记录到扩展异常；阶段剖析的 gather/attention PyTorch fallback 计数为零。正常按有效邻居比例在 packed 与 dense attention 间选择仍保留。计时只留扩展调用计数包装，完整源码计数在剖析打开；两者的开销和口径没有混用。

![FlashTGN 诊断计时与阶段分布](analysis/flash_baseline.png)

**内核追踪把下一目标收窄到 W_kv 的反向矩阵乘法。**

每种配置再运行一轮原版完整训练，用 profiler 在中部只记录 1 个真实训练 step（batch_size=200 为第 277 步，batch_size=2000 为第 29 步，均按一开始）。先有一段真实训练和 profiler warmup，结束时额外同步一次保证 GPU 活动完整落在窗口内。这八个单步仅用于归因，不加入上面的性能样本。

使用 CPU 事件 UID、forward/backward sequence、Chrome flow 和 CUDA correlation 对齐到前向模块；'''+str(k['linked_activities'])+'/'+str(k['total_activities'])+''' 条窗口内 GPU 活动均能归属，残留 backward 归属歧义与两套 CPU 树交叉检查分歧均为零。每个 trace 的直接 LSE launch 没有 External id，使用相同 correlation 的 runtime 记录及唯一最内层 CPU scope 补齐；没有靠内核名称猜测归属。完整覆盖细节和原始 trace 均保留。

|batch_size|邻居数|层数|W_kv 矩阵乘法占该步 backward GPU 活动时长|再含 W_kv bias reduction|
|---:|---:|---:|---:|---:|
'''
for row in k['cases']:
 c=s['timing'][row['label']]['case'];text+=f"|{c['bs']}|{c['k']}|{c['layers']}|{row['wkv_mm_percent']:.1f}%|{row['wkv_mm_bias_percent']:.1f}%|\n"
text+='''
这里的分母是单步 backward CUDA 活动时长之和，不是整轮训练时间；不能把这些百分比直接乘以前一张整轮 scope 表计算承诺收益。单步使用的 attention 分支也不能代表每轮所有分支。小 batch 的 trace 中主机/提交间隙尤其明显，单看 GEMM kernel 加速可能无法等比例缩短训练。

`W_kv` 是 key/value 的线性投影，输出 512 维，输入 684 维。大配置的一处真实反向矩阵形状为 `[512,1813700] × [1813700,684]`，同时还要计算输入梯度和 bias reduction。进一步调查应围绕是否有可避免的 padding/投影工作、packed/dense 策略及这些矩阵的实际效率，而不是笼统把所有 backward 都改写成融合核。

trace 已出现 `cutlass_80_tensorop_s1688gemm...` 内核。结合 NVIDIA CUTLASS 对该 TensorOp/3×TF32 实现的说明，可判断当前已有 Tensor Core 矩阵路径；这不是“尚未使用 Tensor Core”的普通 CUDA 基线。这里没有测指令吞吐或利用率，不能据此承诺换一个 Tensor Core 内核就会更快。[NVIDIA CUTLASS 3×TF32 TensorOp 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/27_ampere_3xtf32_fast_accurate_tensorop_gemm.cu)

接下来应先把明确的 mailbox 写入语义固定，并通过冻结相同 backward 输入来量化源实现自身的数值噪声，建立能区分候选误差与源自重放噪声的验证办法；保留这次原门槛失败结果。然后针对 W_kv 的工作量/布局做一个支付全部代价的小实验，再在相同 FlashTGN 语义和完整配置矩阵上验证整轮收益。本次没有继续开发新优化内核，也没有宣称 1.10× 已达成。

[复现说明](REPRODUCE.md)、[原始合同](CONTRACT.md)、[失败与修订记录](ATTEMPTS.md)、[原版 CPU 审计](evidence/qualify_v1/audit.json)、[写入探针 CPU 审计](evidence/mailbox_probe_v1/audit.json)、[修复版 CPU 审计](evidence/qualify_repaired_v1/audit.json)、[训练末状态审计](analysis/training_audit.json)、[内核归因](analysis/kernel_findings.json)均已保留。大型 tensor 原件保留在远程实验目录，哈希及本地传输范围由 remote_manifest.json、transfer_verification.json 和 MANIFEST.sha256 明确记录。
'''
(r/'验证与性能结果.md').write_text(text)
print(len(text), 'characters')
