"""内置种子提示词：严格按 MiniMax H3 官方两套壳写，并且**自身必须过规则引擎**。

用途
----
1. 插件开箱即用的「正例」来源（用户不提供语料时，靠种子 + 规则降级也能立刻
   造出训练对，先把流水线跑通）；
2. 训练/自测的格式基准；
3. 文档里的示例。

「文生视频」（BASE_SECTIONS）与「多参考图」（REF_SECTIONS）是两套**不同**
字段，绝不能混用 —— 这里分开存放。

写作约束（全部来自 ``wushu_bridge/lint/h3lint.py`` 的实测判据）
--------------------------------------------------------------
* 分镜 **1~3 镜**（4 镜=warn，6 镜以上=error：切太碎）
* ``[Shot 1]`` **不写时间码**，时间码从 ``[Shot 2]`` 开始
* 每个切镜的**开头 170 字**必须交代位置/朝向（在左/在右/面向/间距…）
* 每个切镜要重申锚点（同一张脸 / 同一套服装与武器）
* 正文**不出现** ``【…】``（会被判成残留规则段）
* 正文**不出现**否定式措辞（禁止/不要/不得…）——负面项属于 ComfyUI 的
  negative prompt，不进正向提示词
* 每镜 2~4 句（一拍一句会被判"一镜装不满"）
* 因果连接词 ≥2（随即/因此/于是/趁/被…）
* 开场即交手，不写「对峙」空转
* 不写慢动作/定格/剑气/瞬移/血条 UI

v1.1 起额外覆盖 BUNNY 启发的高动态家族：缴械回收、撞墙反弹、遮挡再识别、
追击刹停再交手、1v2 交接、伤势跨镜继承、湿街追逐（武打味）。
v1.1.2 起：显式 xyz 坐标锁定（角色A@xyz=(-2,0,0) / <Subject 1> at xyz=…）。
v2.1 起：漫剧老李连续/衔接/击打反馈/被击反应（防守三态、机枪对招、动势衔接、被击弧、对撞两段、终结三幕、穿梭换位、段缝末态）。
"""

from __future__ import annotations

from typing import Dict, List

SEED_T2V: List[str] = [
    # ── 1. 对打完整弧：雨夜长街，太刀 vs 单刀（3 镜）──────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 雨夜长街，两侧灯笼暖光，湿石板反光。角色A是黑发披散的男性，黑色武士劲装，双手持野太刀，画面左前方，间距2格。角色B是灰发束髻的男性，靛蓝汉服武袍，右手持雁翎单刀，画面右后方。此刻角色A的刀尖已经点出，角色B的刀刃横在胸前，两把兵器都挂着水光。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot，胸口手持跟拍。角色A后脚蹬湿石板，转腰送胯，踏步把间距拉近到1格，普攻出「刀锋点刺」，刀尖走直线刺向角色B胸口，随即接招式「过肩劈」，太刀借着惯性走完整弧线。角色B举刀斜挡，刃对刃撞出火星，因此被压得后退半步，鞋底在湿石上打滑，架门被压开、失衡未完全恢复。角色A趁这个空档继续前压，两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.100.low angle, tracking，贴地跟拍刀弧。角色B在画面左侧，面向角色A，身位退到2格，侧移半步错开刀路，普攻出「撩刀」，刀自下向上兜起。角色A收刀回架，用刀脊磕住刀身，火星溅起，因此虎口发麻，太刀刀势下沉。角色A随即撤步把间距拉回2格，角色B伤侧发力仍受限，两人仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.600.over-the-shoulder，过角色A肩看角色B。角色A在左侧，朝向角色B，间距2格，拧身换步绕到角色B左前方，招式「斜劈」让太刀斜下走弧。角色B来不及格挡，硬吃一刀，左肩衣料撕裂，血线渗出，因此踉跄倒退2格，重心压在右腿，伤肩仍受限。角色A跟步送刀，终结技「过肩劈」走完整刀路，角色B举刀硬架却被震脱手，于是沿刀的作用线仰面倒地，湿石水花溅起，不再起身。
overall_soundscape: 踏湿石、刀弧破空、兵刃相交火星、衣料撕裂、闷哼、雨声、粗喘。
non_diegetic_music: None.""",
    # ── 2. 枪 vs 棍 ──────────────────────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 黄昏演武场，夯土地面扬尘，右侧木栏为实体。角色A是青布短打的女性，持白蜡长枪，画面左侧，间距3格。角色B是赤膊束腰的男性，持齐眉棍，画面右侧。此刻角色A的枪尖已经指向角色B胸口高度，角色B的棍横在身前。
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow。角色A后脚蹬地，腰胯前送，上步把间距拉近到2格，普攻「扎枪」让枪尖走直线刺向胸口高度。角色B横棍封挡，棍身中段磕住枪杆，火星溅起，因此后退半步泄力，前脚掌搓地扬起尘土。角色A趁枪杆被磕开的角度顺势收枪，两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.000.low angle，低机位看棍梢。角色B在右侧，面向角色A，身位仍在2格，侧移半步绕开枪线，招式「扫棍」扫向膝部，棍走平弧。角色A提膝避过，用枪尾向下压住棍杆，杆身相碰发出闷响，因此被迫换步。角色A随即撤步把间距拉回3格，尘土仍留在两人之间，两人仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.500.斜侧推进，medium shot。角色A在左侧，朝向角色B，间距3格，转身借腰力反手刺出招式「回马枪」。角色B踏步逼进1格缩短长兵器优势，举棍硬挡，棍梢只擦到枪杆，因此被枪尖掠过肋侧，衣角被带起。角色B顺势前压，终结技「劈棍」从肩后走完整弧线砸向肩头，角色A举枪横架却被砸弯枪杆，于是单膝跪地撑在夯土上，棍梢砸地扬起尘土，角色A仍撑着没有倒地。
overall_soundscape: 夯实脚步、枪杆破空、棍身闷响、尘土落地、衣料摩擦、粗喘。
non_diegetic_music: None.""",
    # ── 3. 缴械 + 夺回 ───────────────────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 青砖院落，井沿为实体，地面微潮。角色A是束发青衫男性，双手持长剑，画面左侧，间距2格。角色B是短打劲装女性，右手持短刀，画面右侧。此刻两刃已经架在一起。
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow。角色A后脚蹬砖，转腰送肩，上步收到1格，普攻「直刺」走直线。角色B侧闪半步用刀脊磕开剑锋，刃对刃脆响，因此剑势偏斜。角色A顺势用剑脊压住短刀握把一拧，短刀脱手落地，归属从角色B转到青砖地面，两人仍是同一张脸、同一套服装，角色B此刻空手。
[Shot 2] At 00:02.200.low angle，贴地看落刀。角色B在右侧，面向角色A，间距1格，俯身去捡回短刀；角色A跟步踩住刀脊阻止回收。角色B沉胯卸力抽手，改空手架门，因此握持权暂时仍在地面。角色B随即撤步拉开到2格，两人仍是同一张脸、同一套服装。
[Shot 3] At 00:05.400.over-the-shoulder。角色A在左侧朝向角色B，间距2格，招式「斜劈」压下。角色B侧滚避开，滚势中捡回短刀，于是握持权回到角色B右手。角色B借起身惯性「撩刀」反击，角色A举剑硬架却被震得虎口发麻，于是角色B跟步终结技「点刺」刺中肩窝，角色A沿作用线单膝跪地，不再起身抢攻。
overall_soundscape: 踏砖、兵刃相交、刀落青砖、翻滚刮地、闷哼、粗喘。
non_diegetic_music: None.""",
    # ── 4. 撞墙反弹 + 动量继承 ───────────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 夜巷窄街，两侧土墙为实体，夯土地面。角色A是短打持单刀女性，画面左侧，间距2格。角色B是劲装持短棍男性，画面右侧，背后半格即是土墙。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot。角色A后脚蹬土，转腰送胯，踏步收到1格，招式「过肩劈」走完整垂弧。角色B举棍横挡，棍身中段硬吃一记，因此被击退，后背撞上实体土墙，冲击沿脊柱回传，动量未泄。两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.000.low angle，贴墙跟拍。角色B在右侧贴墙，面向角色A，借墙面反弹前冲，顺势把余势送进「扫棍」扫向膝线，棍走平弧。角色A提膝避过并用刀脊下压，火星溅起，因此被迫换步。角色A随即撤半步，动量继承的余波仍让角色B身位前压，两人仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.200.斜侧推进。角色A在左侧朝向角色B，间距1格，拧腰「横扫」回敬。角色B来不及完整格挡，硬吃刀脊一下踉跄，于是角色A跟步终结技「正劈」砸开架门，角色B沿作用线侧倒撞墙再滑落，不再起身。
overall_soundscape: 踏土、刀风、棍身闷响、撞墙回声、火星、粗喘。
non_diegetic_music: None.""",
    # ── 5. 遮挡后入画再识别 ─────────────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 雨夜长街，右侧灯笼柱为实体掩体，湿石反光。角色A是黑衣持太刀男性，画面左侧，间距2格。角色B是靛蓝袍持单刀男性，画面右侧。
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow。角色A后脚蹬湿石，上步收到1格，普攻「点刺」刺向胸口。角色B斜挡刃对刃出火星，因此侧移把身体没入灯笼柱后方，被掩体短暂遮挡出画，面部与兵器暂时不可见。角色A踩着水花绕柱，湿石仍湿。
[Shot 2] At 00:02.300.tracking，绕柱跟拍。角色A在左侧朝向柱后，间距1格。角色B从灯笼柱另一侧入画，仍是同一张脸、同一套服装与武器，朝向面向角色A，没有左右颠倒串人。角色B借出柱惯性「撩刀」上打，角色A刀脊磕住，因此虎口发麻。两人仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.500.over-the-shoulder。角色A在左侧，角色B在右侧，间距2格。角色A拧身「斜劈」压下，角色B硬吃一刀左肩衣破渗血，伤肩仍受限，因此踉跄。角色A跟步终结技「过肩劈」，角色B脱手仰面倒地，水花溅起，不再起身。
overall_soundscape: 踏湿石、绕柱衣料、兵刃相交、衣裂、雨声、粗喘。
non_diegetic_music: None.""",
    # ── 6. 追击 → 超步 → 刹停再交手 ─────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 青石坡道，两侧矮墙，薄雾。角色A是青衫持剑男性，画面后方追击位。角色B是灰袍持刀男性，画面前方逃跑位，初始间距3格。
integrated_multimodal_description:
[Shot 1] tracking, medium shot，跟追。角色A后脚蹬青石追击，把间距从3格压缩到2格，转腰「直刺」探出。角色B边跑边回头斜挡，刃面擦过火星，因此身位继续前窜。两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.100.low angle，贴地看脚步。角色A外侧超步越过角色B肩线，朝向短暂同向，随即蹬地刹停回身，重新对面向角色B举剑。角色B刹步回防举刀，间距回到1格，因此两人重新对位交手。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.300.over-the-shoulder。角色A在左侧面向角色B，间距1格，招式「斜劈」打进。角色B格挡不及硬吃，衣料撕裂，于是角色A跟步终结技「点刺」刺中肋侧，角色B沿作用线跪倒青石，不再起身。
overall_soundscape: 追步踏石、刹停搓地、兵刃相交、衣裂、粗喘、薄雾风声。
non_diegetic_music: None.""",
    # ── 7. 1v2 交接 / 排序 ───────────────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 夜巷岔口，夯土路面，两侧土墙。角色A是短打持单刀女性，开场居中。角色B是劲装持短棍男性，在角色A左前方1格。角色C是布衣持竹矛男性，在角色A右后方2格。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot。角色A正对角色B，暂时把角色C压在视野边缘，先结算与角色B的这一拍：后脚蹬地「正劈」垂弧劈中角色B肩头，因此角色B侧倒出画。攻击权随即交接，角色A回身面向角色C。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.000.tracking。角色C从右后方上步接手，间距收到1格，竹矛「扎枪」刺来。角色A侧闪用刀脊磕开矛杆，火星与木屑溅起，因此角色C前冲失衡。角色A明确下一拍目标是角色C而不是已倒地的角色B，两人（A与C）仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.000.wide shot。角色A在左侧朝向角色C，跟步终结技「横扫」走平弧扫中腰侧，角色C沿作用线倒地，不再起身。本场攻击排序：先B后C，交接清楚。
overall_soundscape: 踏土、刀风、棍倒、矛杆磕击、倒地、夜风、粗喘。
non_diegetic_music: None.""",
    # ── 8. 湿街追逐（武打味高动态行为：归属+朝向+环境）──────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 雨夜湿石长街，灯笼暖光，地面持续反光仍湿。角色A是黑衣持短刀男性，画面后方。角色B是灰衣持包袱与腰刀男性，画面前方，间距4格。
integrated_multimodal_description:
[Shot 1] tracking, medium shot。角色A蹬湿石追击压缩到2格，角色B回身抽腰刀格挡，刃对刃出火星，因此包袱带滑、包袱仍由角色B左手握持，归属明确。鞋底打滑，湿石仍湿未重置。两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.200.low angle。角色A外侧超步，朝向短暂同向，随即刹停回身重新面向角色B。角色B换到左侧、角色A在右侧，间距1格，相对朝向锁定，没有左右串人。角色B顺势「撩刀」反击，角色A刀脊架住。
[Shot 3] At 00:05.400.over-the-shoulder。角色A在右侧面向左，角色B在左侧面向右。角色A拧腰「斜劈」打掉角色B架门，腰刀脱手落地，握持权落到湿石；包袱仍在角色B左手。角色A跟步逼得角色B单膝跪地撑住，于是追击结束，角色B不再起跑。
overall_soundscape: 踏湿石打滑、追步、兵刃相交、包袱布料、雨声、粗喘。
non_diegetic_music: None.""",
    # ── 9. 面对面交手（朝向锁，防背对空砍）──────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 青砖演武场，正午硬光。角色A是青衫持剑男性，画面左侧。角色B是灰袍持刀男性，画面右侧。两人开场已面对面，刀尖相抵。
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow。坐标锁定：角色A@xyz=(-2,0,0) 与 角色B@xyz=(2,0,0) 面对面，间距2格，刀尖指向对方胸口。角色A后脚蹬砖转腰，「直刺」朝对方刺出。角色B举刀斜挡，刃对刃出火星，因此后退半步，两人始终面对面。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.200.over-the-shoulder。接上一镜：角色A@xyz=(-2,0,0) 仍朝向 角色B@xyz=(2,0,0)，间距1格，切镜后重申相同 xyz，没有背对。角色B「撩刀」朝对方反击，角色A刀脊磕住，因此虎口发麻。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.400.斜侧推进。角色A在左侧面对面朝向角色B，间距1格，终结技「斜劈」朝对方肩线压下，角色B硬吃踉跄，于是沿作用线单膝跪地，不再起身。
overall_soundscape: 踏砖、兵刃相交、闷哼、粗喘。
non_diegetic_music: None.""",
    # ── 10. 有因跳跃：蹬地起跳 + 落地卸力 ────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 夜巷，夯土地面，土墙为实体。角色A是短打持单刀女性，画面左侧，间距2格。角色B是劲装持棍男性，画面右侧。
integrated_multimodal_description:
[Shot 1] medium shot, low angle。角色A在左侧面向角色B，后脚蹬土借力起跳，腰胯上送，双脚离地；腾空中面向角色B走「过肩劈」下劈，刃口对准对方肩线。角色B举棍斜挡，棍身闷响。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.000.tracking。接上一镜：角色A落地屈膝卸力，冲击散入夯土，站稳后再跟步收到1格，没有无意义悬空。角色B在右侧面向角色A，趁落地瞬间「扫棍」扫膝，角色A提膝避开。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.200.over-the-shoulder。角色A在左侧朝向角色B，终结技「横扫」走平弧，角色B硬吃侧倒，于是不再起身。
overall_soundscape: 蹬土起跳、衣料破空、落地踏实、棍身闷响、粗喘。
non_diegetic_music: None.""",
    # ── 11. 法术瞄准对手 + 击中反馈 ─────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 石殿内，地面干燥，两侧石柱为实体。角色A是道袍持短杖男性，画面左侧。角色B是劲装持剑男性，画面右侧。开场已面对面。
integrated_multimodal_description:
[Shot 1] medium shot, handheld。角色A在左侧与角色B面对面，间距3格，抬手对准角色B胸口蓄力。角色A后脚蹬地沉胯，掌力法术射向角色B胸口，弹道可追。角色B侧闪半步，掌力擦肩而过溅起石屑。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.300.tracking。接上一镜：角色A仍在左侧朝向角色B，间距2格。角色A再次施法，掌力瞄准角色B胸口击中，衣料震起灼痕，因此角色B踉跄倒退1格，闷哼一声，重心不稳。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.500.over-the-shoulder。角色A跟步面对面逼近，短杖终结技点向肩窝，角色B沿作用线单膝跪地，于是不再起身。
overall_soundscape: 掌力破空、石屑、衣料灼响、闷哼、粗喘。
non_diegetic_music: None.""",
    # ── 12. 切镜身份+空间锁（无瞬移无换人）─────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 雨夜长街，湿石反光，灯笼柱为实体。角色A是黑衣持太刀男性，黑发披散。角色B是靛蓝袍持单刀男性，灰发束髻。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot。坐标锁定：角色A@xyz=(-2,0,0) 面向 角色B@xyz=(2,0,0)，间距2格。角色A蹬湿石「点刺」朝对方刺出，角色B斜挡出火星，因此退半步。两人仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.100.low angle。接上一镜：角色A@xyz=(-2,0,0) 仍朝向 角色B@xyz=(2,0,0)，间距1格，切镜后重申相同 xyz，位置连续没有瞬移。角色B「撩刀」朝对方反击，角色A刀脊磕住。仍是同一张脸、同一套服装与武器，黑发与靛蓝袍未变人。
[Shot 3] At 00:05.400.over-the-shoulder。接上一镜：角色A在左侧，角色B在右侧，间距1格，朝向面对面。角色A终结技「过肩劈」朝对方压下，角色B硬吃仰面倒地，于是不再起身。湿石仍湿。
overall_soundscape: 踏湿石、兵刃、雨声、闷哼、粗喘。
non_diegetic_music: None.""",

    # ── 13. XYZ 坐标锁定对决（中文）──────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 青砖演武场，硬光。角色A是青衫持剑男性。角色B是灰袍持刀男性。
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow。坐标锁定：角色A@xyz=(-2,0,0) 面向 角色B@xyz=(2,0,0)，间距2格，刀尖相抵。角色A后脚蹬砖转腰，「直刺」朝对方刺出。角色B举刀斜挡，刃对刃出火星，因此后退半步，两人始终面对面。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.200.over-the-shoulder。接上一镜：角色A@xyz=(-2,0,0) 仍面向 角色B@xyz=(2,0,0)，间距1格，切镜后重申相同 xyz，没有瞬移。角色B「撩刀」朝对方反击，角色A刀脊磕住，因此虎口发麻。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.400.斜侧推进。角色A上步，xyz 更新为(-1,0,0)；角色B撤步至(1,0,0)，仅在位移动作后更新坐标。终结技「斜劈」朝对方肩线压下，角色B硬吃踉跄，于是沿作用线单膝跪地，不再起身。
overall_soundscape: 踏砖、兵刃相交、闷哼、粗喘。
non_diegetic_music: None.""",

    # ── 14. XYZ coords locked duel (EN) ───────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. Brick yard, hard noon light. Fighter A in cyan robe with sword. Fighter B in grey robe with saber.
integrated_multimodal_description:
[Shot 1] medium shot, handheld follow. coords locked: Fighter A@xyz=(-2,0,0) facing Fighter B@xyz=(2,0,0) at two steps, tips touching. A drives the rear foot into a straight thrust. B parries; sparks fly, so B yields half a step. They stay face to face. Same faces, same costumes and weapons.
[Shot 2] At 00:02.200.over-the-shoulder. Continuing: Fighter A@xyz=(-2,0,0) still facing Fighter B@xyz=(2,0,0) at one step — restate same xyz after the cut; no teleport. B flicks upward; A jams with the spine, grip goes numb. Same faces, same costumes and weapons.
[Shot 3] At 00:05.400.oblique push-in. A steps in, xyz updates to (-1,0,0); B retreats to (1,0,0) — update only after explicit footwork. Finisher diagonal cut; B takes it, drops to one knee along the line of force, and does not get up.
overall_soundscape: brick feet, steel clash, grunt, breath.
non_diegetic_music: None.""",

    # ── 15. 防守三态 + 被击反应（崩防→狼狈）──────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 青砖演武场，硬光。角色A是短打持拳男性，画面左侧。角色B是劲装持高护架男性，画面右侧。开场已面对面。
integrated_multimodal_description:
[Shot 1] medium close-up, handheld。坐标锁定：角色A@xyz=(-2,0,0) 与 角色B@xyz=(2,0,0) 面对面，间距2格。角色B桩架稳固，双臂架桥护头护胸，完整态格挡。角色A垫步逼近，后脚蹬地转腰，右直拳砸向架门。仍是同一张脸、同一套服装。
[Shot 2] At 00:02.200.近景。接上一镜：角色A@xyz=(-1,0,0) 仍朝向角色B，重击砸开架门，因此角色B崩防线：双臂向两侧震开，中线大开半拍，眉骨被擦中溅火星，重心不稳。角色B来不及完整格挡，只能侧身避半拍。仍是同一张脸、同一套服装。
[Shot 3] At 00:05.400.斜侧。于是角色B狼狈过渡：极低下潜贴地再碎步撤步后撤，本能单臂乱拍换架不及；角色A趁这个空档跟步左横肘切肋，衣料闷响，角色B硬吃身躯弓成虾形沿作用线单膝跪地，伤势仍受限，不再起身满血架防。
overall_soundscape: 踏砖、拳风、肘击闷响、火星、闷哼、粗喘。
non_diegetic_music: None.""",
    # ── 16. 近景机枪对招 + 肘击序列 ─────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 夜巷夯土，两侧土墙。角色A是短打女性，角色B是劲装男性，上半身近景对招。
integrated_multimodal_description:
[Shot 1] medium close-up, handheld follow。角色A在左侧面对面朝向角色B，间距1格，垫步逼近。机枪节奏：右摆拳打脸→左勾拳打下颌→右直拳打鼻梁→左横肘切眉骨；角色B依次格挡、侧身避、下潜闪、拍开换架。仍是同一张脸、同一套服装。
[Shot 2] At 00:02.100.近景。接上一镜动作中段续：肘尖切中眉骨溅火星，因此角色B踉跄闷哼，崩防线中线大开，重心不稳。角色A趁这个空档踏步跟步补膝撞肋，环形微气爆荡开衣料。仍是同一张脸、同一套服装。
[Shot 3] At 00:05.200.over-the-shoulder。于是角色A终结技顶肘贯心口，角色B硬吃身躯凹陷弓形，退半步后倒退嵌向土墙，地面扬尘，不再起身。
overall_soundscape: 拳肘破空、拍架、火星、闷响、撞墙、粗喘。
non_diegetic_music: None.""",
    # ── 17. 动势衔接切 + 段缝末态 ───────────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 雨夜湿石长街，灯笼柱实体。角色A黑衣持太刀，角色B靛蓝袍持单刀。
integrated_multimodal_description:
[Shot 1] medium shot, tracking。角色A在左侧面向角色B，间距2格，垫步逼近，后脚蹬湿石转腰，「过肩劈」挥到过半，弧线未完，刀尖尚未触及；角色B举刀格挡蓄势。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.000.特写切中景。接上一镜动作中段续：同一刀势补完弧线命中角色B肩线，刃面啃入衣料溅火星与微气爆，因此角色B硬吃踉跄，伤肩衣破渗血，重心不稳。角色B碎步撤步泄力，换机位续打。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.300.over-the-shoulder。接上一镜末态起手：角色B单膝跪于湿石右侧，刀尖垂地，伤肩仍受限；角色A在左侧持刀高位未收。于是角色A趁这个空档终结技「斜劈」压下，角色B狼狈残架被震开，仰面倒地，湿石仍湿，不再起身。
overall_soundscape: 踏湿石、刀弧、命中火星、衣裂、雨声、粗喘。
non_diegetic_music: None.""",
    # ── 18. 对撞两段 + 穿梭换位 + 终结三幕 ─────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 焦土战场，碎石。角色A银白劲装持剑，角色B黑红劲装持刀。
integrated_multimodal_description:
[Shot 1] wide oblique, tracking。角色A自左向右冲锋、角色B自右向左对冲，间距从3格拉到交叉点，画面中央交叉换位擦肩，刃面互刮溅火星，朝向短暂同向；双方踏步碾步不停。仍是同一张脸、同一套服装与武器。
[Shot 2] At 00:02.200.中景。于是两人回身硬撞：接触点爆闪与环形气浪，地面放射状龟裂；双向反震分离各退半步，立刻垫步接新动作禁止僵持，角色B举刀格挡不及。仍是同一张脸、同一套服装与武器。
[Shot 3] At 00:05.400.仰拍跟进。破局：角色A趁这个空档重击砸开角色B双臂崩防线；绽放：踏步凌空下劈终结技全力砸落；余震：冲击波扩散，角色B硬吃弓形倒飞，碎石滞空，终帧定格最炸裂瞬间，不再起身。
overall_soundscape: 冲刺、刃刮、对撞气浪、地裂、冲击波、粗喘。
non_diegetic_music: None.""",

]

SEED_REF2V: List[str] = [
    # ── 多参考图：雨夜对决（原有，略加强状态继承）────────────────────
    """subject_definitions:
<Subject 1> is the man in <Picture 1>, with loose black hair, a black warrior's jacket, and a long tachi held in both hands. Preserve his face, hairstyle, clothing, and body proportions.
<Subject 2> is the man in <Picture 2>, with tied-back grey hair, an indigo han-style martial robe, and a single yanling saber in his right hand. Preserve his face, hairstyle, clothing, and body proportions.
<Subject 3> is the rainy night street in <Picture 3>, with warm lantern light on both sides and wet reflective stone slabs, and a solid lantern post on the right.
summary:
[reference generation] A 10-second rainy-night duel in which <Subject 1> presses <Subject 2> from two steps to one, and finishes him with an overhead chop that puts him on the ground.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, loose black hair, black jacket, and tachi remain unchanged.
<Subject 2> (appears in [Shot 2]): fully_preserved - face, grey tied-back hair, indigo robe, and saber remain unchanged.
<Subject 3> (appears in [Shot 3]): fully_preserved - lantern positions, wet slabs, and the lantern post stay stable.
detailed_description:
[Shot 1] handheld follow, medium shot from chest height. <Subject 1> drives his rear foot into the wet stone, turns his waist, and steps in from two steps to one, then thrusts with the tachi tip along a straight line and follows with an overhead chop that carries the blade through a full arc. <Subject 2> raises his saber into a diagonal guard; edge meets edge, sparks fly, and he is pushed back half a step with his sole sliding on the wet stone, guard forced open and balance not fully recovered. <Subject 1> presses the opening, and the two are still the same two fighters, same faces, same costumes and weapons.
[Shot 2] At 00:02.100.low angle, tracking, following the blade arc close to the ground. <Subject 2> is on the left, facing <Subject 1>, holding two steps of distance, and slides half a step to the side to clear the blade path, then flicks his saber upward. <Subject 1> pulls the tachi back into a guard and jams the blade with his spine; sparks scatter and his grip goes numb, so the tachi sinks. <Subject 1> then steps back to two steps of distance; <Subject 2>'s injured side stays limited, and the two are still the same two fighters, same faces, same costumes and weapons.
[Shot 3] At 00:05.600.over-the-shoulder from behind <Subject 1>. <Subject 1> is on the left, facing <Subject 2> at two steps, turns his hips and steps around to <Subject 2>'s front-left, and cuts diagonally downward. <Subject 2> cannot guard in time and takes the cut; the cloth on his left shoulder tears and a thin line of blood shows, so he staggers back two steps with his weight on the right leg, injured shoulder still limited. <Subject 1> follows through with a finishing overhead chop along the full arc; <Subject 2> braces with his saber but it is knocked from his hands, and he falls onto his back along the line of force, water splashing on the wet stone, and does not get up.
overall_soundscape: feet on wet stone, blade arcs cutting air, steel on steel with sparks, cloth tearing, a muffled grunt, rain, heavy breathing.
non_diegetic_music: None.""",
    # ── 新：墙体反弹 + 武器回收（ref2v）──────────────────────────────
    """subject_definitions:
<Subject 1> is the woman in <Picture 1>, short martial jacket, single saber in the right hand. Preserve face, hair, clothing, proportions.
<Subject 2> is the man in <Picture 2>, sleeveless top, short staff in both hands. Preserve face, hair, clothing, proportions.
<Subject 3> is the night alley in <Picture 3>, earthen walls as solid cover, packed-earth ground.
summary:
[reference generation] A 10-second alley duel: <Subject 1> knocks <Subject 2> into the wall, he rebounds with inherited momentum, loses the staff, then reclaims it before the finisher.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, jacket, saber unchanged.
<Subject 2> (appears in [Shot 2]): fully_preserved - face, sleeveless top, staff ownership tracked across drop and reclaim.
<Subject 3> (appears in [Shot 3]): fully_preserved - walls and ground marks persist.
detailed_description:
[Shot 1] handheld follow, medium shot. <Subject 1> drives the rear foot, turns the waist, and closes from two steps to one with an overhead chop. <Subject 2> braces the staff across; the impact knocks him back into the solid earthen wall, momentum not yet spent, so he rebounds forward riding the residual force. They are still the same two fighters, same faces, same costumes and weapons.
[Shot 2] At 00:02.200.low angle near the wall. <Subject 2> is on the right facing <Subject 1>, rides the wall rebound into a low sweep; <Subject 1> lifts a knee and jams with the saber spine, sparks fly, and the staff is knocked free — ownership leaves <Subject 2> to the packed earth. <Subject 2> stoops and reclaims the staff, so ownership returns to his hands; they are still the same two fighters, same faces, same costumes and weapons.
[Shot 3] At 00:05.400.over-the-shoulder. <Subject 1> on the left faces <Subject 2> at one step, cuts diagonally; <Subject 2>'s guard is late after the stoop, takes the hit, and <Subject 1> finishes with a vertical chop along the line of force. <Subject 2> falls and does not get up; debris and wall scuff from the rebound stay in frame.
overall_soundscape: feet on packed earth, staff thud on wall, steel on wood sparks, grunt, night wind.
non_diegetic_music: None.""",
    # ── 新：面对面 + 法术瞄准击中反馈 + 切镜身份空间锁（EN ref2v）──
    """subject_definitions:
<Subject 1> is the man in <Picture 1>, daoist robe, short staff in the right hand. Preserve face, hair, clothing, proportions.
<Subject 2> is the man in <Picture 2>, martial jacket, straight sword. Preserve face, hair, clothing, proportions.
<Subject 3> is the stone hall in <Picture 3>, dry floor, solid pillars on both sides.
summary:
[reference generation] A 10-second face-to-face duel: <Subject 1> casts palm-force aimed at <Subject 2>'s chest, the hit staggers him with a scorch mark, then a staff finisher ends the fight without teleport or identity drift.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, robe, staff unchanged.
<Subject 2> (appears in [Shot 2]): fully_preserved - face, jacket, sword unchanged.
<Subject 3> (appears in [Shot 3]): fully_preserved - pillars and floor marks persist.
detailed_description:
[Shot 1] handheld follow, medium shot. <Subject 1> on the left stands face to face with <Subject 2> on the right at three steps, raises a hand aimed at <Subject 2>'s chest, drives the rear foot and casts palm-force toward <Subject 2>'s chest. <Subject 2> sidesteps; the bolt grazes the pillar with stone chips. They are still the same two fighters, same faces, same costumes and weapons.
[Shot 2] At 00:02.300.tracking. Continuing: <Subject 1> still on the left facing <Subject 2> at two steps — no teleport. He casts again aimed at <Subject 2>'s chest; the spell hits, cloth jolts with a scorch mark, so <Subject 2> staggers back one step with a grunt, off-balance. Same faces, same costumes and weapons.
[Shot 3] At 00:05.500.over-the-shoulder. <Subject 1> on the left faces <Subject 2> at one step, finishes with a staff tip to the shoulder hollow; <Subject 2> drops to one knee along the line of force and does not get up.
overall_soundscape: palm-force whoosh, stone chips, cloth scorch, grunt, breath.
non_diegetic_music: None.""",


    # ── Ref2V: XYZ coords locked face-to-face (EN) ───────────────────
    """subject_definitions:
<Subject 1> is the man in <Picture 1>, cyan robe, straight sword. Preserve face, hair, clothing, proportions.
<Subject 2> is the man in <Picture 2>, grey robe, saber. Preserve face, hair, clothing, proportions.
<Subject 3> is the brick yard in <Picture 3>, hard noon light, solid ground.
summary:
[reference generation] A 10-second coords-locked duel: <Subject 1> at xyz=(-2,0,0) faces <Subject 2> at xyz=(2,0,0); after cuts the same xyz is restated; footwork alone updates coords.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, robe, sword unchanged.
<Subject 2> (appears in [Shot 2]): fully_preserved - face, robe, saber unchanged.
<Subject 3> (appears in [Shot 3]): fully_preserved - yard and light persist.
detailed_description:
[Shot 1] handheld follow, medium shot. coords locked: <Subject 1> at xyz=(-2,0,0) facing <Subject 2> at xyz=(2,0,0) at two steps. <Subject 1> drives the rear foot into a thrust; <Subject 2> parries with sparks and yields half a step. Same faces, same costumes and weapons.
[Shot 2] At 00:02.200.over-the-shoulder. Continuing: <Subject 1> at xyz=(-2,0,0) still facing <Subject 2> at xyz=(2,0,0) — restate same xyz after the cut; no teleport. <Subject 2> counters; spine jam. Same faces, same costumes and weapons.
[Shot 3] At 00:05.400.oblique push-in. <Subject 1> steps in, xyz updates to (-1,0,0); <Subject 2> retreats to (1,0,0) — update only after explicit footwork. Finisher drops <Subject 2> to one knee; he does not get up.
overall_soundscape: brick feet, steel, grunt, breath.
non_diegetic_music: None.""",
    # ── 新：防守三态 + 被击反应 + 动势衔接（ref2v）──────────────────
    """subject_definitions:
<Subject 1> is the woman in <Picture 1>, short martial jacket, bare fists. Preserve face, hair, clothing, proportions.
<Subject 2> is the man in <Picture 2>, martial top, high guard. Preserve face, hair, clothing, proportions.
<Subject 3> is the brick yard in <Picture 3>, hard noon light, solid ground.
summary:
[reference generation] A 10-second close exchange: <Subject 1> breaks <Subject 2>'s full guard through defense 3-states, then finishes with an elbow after match-on-action continuity.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, jacket unchanged.
<Subject 2> (appears in [Shot 2]): fully_preserved - face, guard collapse and injury carry tracked.
<Subject 3> (appears in [Shot 3]): fully_preserved - brick marks persist.
detailed_description:
[Shot 1] medium close-up, handheld. <Subject 2> holds a solid full guard, arms bridged, face to face with <Subject 1> at two steps. <Subject 1> drives the rear foot into a straight cross that smashes the bridge. Same faces, same costumes.
[Shot 2] At 00:02.200.close push-in. Continuing mid-action: the same punch finishes; <Subject 2>'s defense line breaks, arms blown open, center line exposed; brow grazed with sparks, so he staggers. Same faces, same costumes.
[Shot 3] At 00:05.400.over-the-shoulder. Too late to re-guard, messy transition — <Subject 2> dives flat then scrambles; <Subject 1> follows with a horizontal elbow to the ribs, cloth thuds, shrimp-bend along the line of force, drops to one knee and does not reset to a full-health guard.
overall_soundscape: brick feet, punch whoosh, elbow thud, sparks, grunt, breath.
non_diegetic_music: None.""",

]

SEED_HORDE: List[str] = [
    # ── 一打多：网格结算后的分镜（3 镜）──────────────────────────────
    """wushu_action, 10.2 seconds, 243 frames, 16:9, 24fps, 832x480. 夜巷窄街，夯土地面，两侧土墙为实体。角色A是短打劲装持单刀的女性，开场居中，格(10,4)。战场20格宽8格高，可见目标约12个。
integrated_multimodal_description:
[Shot 1] handheld follow, medium shot。角色A后脚蹬地，踏步向前1格到格(9,4)，正前2格有两个目标，普攻「正劈」让刀走垂弧，劈中左侧目标肩头，因此目标侧倒出画，本秒击倒1个。角色A趁刀势回弹拧腰回身，仍是同一人、同一套服装与武器。
[Shot 2] At 00:01.000.low angle，贴地跟刀。右侧贴身一个目标，角色A在格(9,4)朝向格(10,4)，拧腰发力出招式「回身旋转360挥砍」，刀走平圆扫中目标腰侧，因此目标倒退撞墙，本秒击倒1个。角色A随即收刀回到中位，攻击权准备交接向下一个目标，仍是同一人、同一套服装与武器。
[Shot 3] At 00:02.000.wide shot 拉远交代位置。角色A在格(9,4)，格(5,4)附近圈内3个目标，蹬土墙借力跳4格落到格(5,4)，落地屈膝卸力后踏步出招式「横扫」走平弧，连续扫中2个目标，因此两个目标同时倒地。角色A紧接着上步逼近最后1个目标，终结技「过肩劈」走完整刀路，目标沿作用线倒地，不再起身，本场共击倒5个。
overall_soundscape: 踏土、刀风、兵刃入体闷响、倒地、喘息、夜风。
non_diegetic_music: None.""",
]

SEEDS: Dict[str, List[str]] = {
    "t2v": SEED_T2V,
    "ref2v": SEED_REF2V,
    "horde": SEED_HORDE,
}


def all_seeds(mode: str = "t2v") -> List[str]:
    """按模式取种子。``t2v`` 模式会带上 horde 样例（同样是文生视频壳）。"""
    if mode == "ref2v":
        return list(SEED_REF2V)
    return list(SEED_T2V) + list(SEED_HORDE)
