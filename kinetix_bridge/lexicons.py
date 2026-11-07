"""武打语义词表（中英双语）。

这份词表是「语义逻辑翻译」的判据来源：哪些词代表力线、格距、防守反应、
打击反馈、招式、运镜；哪些词是逻辑漏洞（瞬移、无借力腾空、站桩对望）；
哪些词会让 H3 读不懂或稀释动作（规则说明、空泛形容、慢动作、特效剑气）。

**为什么必须双语**：用户的 924 条同分布语料
（``dataset\\h3_train\\metadata.csv``）是**英文为主**的，英文词表就是从那批
语料里统计出来的高频领域词（weight / stance / hips / momentum / gravity /
footwork / parry / thrust / whoosh / rustle …）。桥要学到的是"往训练分布
的那套写法靠"，所以判据必须覆盖那套写法。

中文侧的检查词表以 ``lint/h3lint.py``（h3lint.js 的 Python 移植）为权威版本，
本文件只补充**降级/改写**用的正向槽位词表，避免两套常量互相打架。
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

CJK_RE = re.compile(r"[\u3400-\u9fff]")


def is_chinese(text: str, threshold: float = 0.02) -> bool:
    """粗略判断一段文本以中文为主。"""
    if not text:
        return False
    cjk = len(CJK_RE.findall(text))
    return (cjk / max(1, len(text))) >= threshold


# ── 力线 / 发力链 ──────────────────────────────────────────────────────
FORCE_CHAIN_ZH: List[str] = [
    "蹬地", "蹬后脚", "后脚蹬", "转腰", "拧腰", "腰胯", "沉胯", "坐胯",
    "力从地起", "重心前压", "重心下沉", "沉肩", "坠肘", "送肩", "送胯",
    "顺步", "跟步", "上步", "撤步", "落步", "半步", "借力", "卸力",
    "惯性", "余势", "收势", "回劲",
]
FORCE_CHAIN_EN: List[str] = [
    "weight", "center of gravity", "centre of gravity", "gravity", "hip", "hips",
    "waist", "rear leg", "back leg", "front leg", "back foot", "front foot",
    "drives from the ground", "drives off", "momentum", "explosive",
    "transfer of weight", "transfers weight", "shifts weight", "shifting weight",
    "pivots on the heel", "rotation", "rotational", "stored power", "winding up",
    "wind-up", "wind up", "ground reaction", "planted foot",
]

# ── 空间 / 格距 ────────────────────────────────────────────────────────
DISTANCE_ZH: List[str] = [
    "间距", "格距", "两格", "一格", "三格", "半格", "距离", "贴身", "近身",
    "拉近", "拉开", "退出", "退开", "逼近", "卡位", "站位", "身位",
    "左前方", "右后方", "斜侧", "正面", "背后", "高低位",
]
DISTANCE_EN: List[str] = [
    "distance", "spacing", "closes the distance", "close range", "mid-range",
    "out of range", "within range", "gap", "steps back", "step back",
    "steps forward", "advances", "closes in", "retreats", "backpedals",
    "off-balance distance", "stance width", "left side", "right side",
    "upper left", "lower right", "diagonal", "opposite side",
]

# ── 防守反应 ───────────────────────────────────────────────────────────
DEFENSE_ZH: List[str] = [
    "格挡", "格开", "架住", "封挡", "横挡", "斜挡", "举刀", "举剑", "抬臂",
    "闪避", "侧闪", "后仰", "低头", "侧身", "下潜", "翻滚", "侧滚", "前滚",
    "漏防", "硬吃", "挡不住", "来不及", "卸力", "滑步", "换步",
]
DEFENSE_EN: List[str] = [
    "guard", "guards", "blocks", "block", "blocking", "parry", "parries",
    "parried", "deflect", "deflects", "deflection", "evade", "evades",
    "evasion", "dodge", "dodges", "sidestep", "side-step", "rolls", "roll",
    "rolls low", "sway-back", "leans back", "ducks", "ducks under",
    "slips the strike", "catches the strike", "intercepts", "braces",
    "takes the hit", "fails to guard", "guard is broken", "guard opens",
]

# ── 接触点 / 命中判定 ──────────────────────────────────────────────────
CONTACT_ZH: List[str] = [
    "刃对刃", "刀脊", "刀背", "刀刃", "剑锋", "枪尖", "棍梢", "拳面", "肘尖",
    "接触点", "命中", "劈中", "刺中", "扫中", "磕中", "击中", "未命中",
    "擦过", "蹭过", "贴着", "掠过", "劈空", "落空", "打空", "刃面朝向",
]
CONTACT_EN: List[str] = [
    "impact", "impacts", "contact point", "connects", "clashes", "clang",
    "edge-on-edge", "blade meets blade", "blade bites", "lands cleanly",
    "grazes", "glancing", "skim", "misses", "whiffs", "cuts through empty air",
    "stops short", "deflected off", "hits the shoulder", "strikes the leg",
]

# ── 打击反馈 / 物理后果 ────────────────────────────────────────────────
FEEDBACK_ZH: List[str] = [
    "火星", "溅起", "碎屑", "闷响", "脆响", "颤", "震得", "发麻", "虎口",
    "踉跄", "趔趄", "倒退", "后仰", "单膝", "跪地", "半跪", "扶墙", "撑地",
    "衣料撕裂", "衣破", "血线", "渗血", "汗珠", "喘息", "粗喘", "闷哼",
    "脚下滑", "打滑", "重心不稳", "失去平衡", "倒地", "仰面", "侧倒",
]
FEEDBACK_EN: List[str] = [
    "sparks", "thud", "clang", "shudder", "jarred", "numb", "stagger",
    "staggers", "stumbles", "recoil", "recoils", "slides", "skids", "friction",
    "dust", "snow sprays", "cloth tears", "fabric rips", "blood", "sweat",
    "exhale", "sharp exhale", "grunt", "breath", "kneels", "drops to one knee",
    "loses balance", "off-balance", "falls", "collapses", "sprawls",
    "does not get up", "can't get up", "pinned",
]

# ── 招式名 ─────────────────────────────────────────────────────────────
MOVE_NAMES_ZH: List[str] = [
    "过肩劈", "斜劈", "正劈", "横扫", "突刺", "直刺", "撩刀", "挂刀", "点刺",
    "缠头裹脑", "劈剑", "撩剑", "点剑", "扎枪", "拦枪", "回马枪",
    "扫堂腿", "鞭腿", "侧踹", "直拳", "崩拳", "摆拳", "勾拳",
    "过肩摔", "别腿", "抱摔", "封喉手", "撩阴腿",
]
MOVE_NAMES_EN: List[str] = [
    "overhead chop", "overhead strike", "vertical chop", "diagonal slash",
    "horizontal sweep", "side sweep", "thrust", "reverse thrust", "backhand",
    "reverse-backhand", "upward flick", "liao", "spinning strike", "spinning back",
    "roundhouse", "side kick", "thrust kick", "palm strike", "straight punch",
    "hook punch", "uppercut", "cleave", "downward cut", "rising cut",
    "shoulder throw", "hip throw", "sweeping kick", "low sweep",
]

# ── 身法 ───────────────────────────────────────────────────────────────
FOOTWORK_ZH: List[str] = [
    "踏步", "上步", "撤步", "绕步", "侧移", "滑步", "换步", "错步",
    "垫步", "转身", "回身", "拧身", "沉身",
]
FOOTWORK_EN: List[str] = [
    "shuffle step", "slide step", "sliding step", "pivot", "pivots", "sidestep",
    "side-step", "crossover step", "crossover", "advance step", "back step",
    "circles", "circling", "turns", "turning", "spins", "spinning",
    "lands", "landing", "steps in", "steps out",
]

# ── 运镜（英文机位 + 中文说明）─────────────────────────────────────────
CAMERA_ZH: List[str] = [
    "跟拍", "中景", "全景", "低机位", "过肩", "侧向平移", "推近", "拉远",
    "短弧侧绕", "斜侧推进", "刃面特写", "手持",
]
CAMERA_EN: List[str] = [
    "wide shot", "medium shot", "close-up", "low-angle", "high-angle",
    "overhead shot", "handheld", "tracking", "dolly", "push in", "pull back",
    "over-the-shoulder", "pan", "tilt", "crane", "camera", "frame",
]

# ── 音景 ───────────────────────────────────────────────────────────────
SOUNDSCAPE_ZH: List[str] = [
    "踏湿石", "踏石", "破空", "刀风", "兵刃相交", "火星", "翻滚刮地",
    "衣裂", "闷哼", "雨", "粗喘", "喘息", "夜风", "脚步声",
]
SOUNDSCAPE_EN: List[str] = [
    "whoosh", "clang", "clangs", "thud", "footsteps", "rustling", "rustle",
    "growl", "exhale", "breath", "wind", "scrape", "crack", "crunch",
]

# ── 武器归属 / 握持状态（BUNNY ownership）────────────────────────────
OWNERSHIP_ZH: List[str] = [
    "握持", "双手持", "右手持", "左手持", "握把", "脱手", "掉落", "落地",
    "捡回", "夺回", "换手", "递刀", "递兵", "归属", "握持权", "踩住刀脊",
    "刀脱手", "兵器脱手", "空着手", "徒手", "空手架",
]
OWNERSHIP_EN: List[str] = [
    "holds", "holding", "grip", "grips", "ownership", "drops", "dropped",
    "disarmed", "loses his weapon", "loses her weapon", "reclaims", "reclaim",
    "recovers the", "picks up", "stoops", "empty-hand", "empty hand",
    "hilt-out", "weapon changes hands", "hands off the weapon",
]

# ── 遮挡 / 身份再识别（occlusion re-ID）───────────────────────────────
OCCLUSION_ZH: List[str] = [
    "遮挡", "被遮", "出画", "入画", "仍是同一张脸", "同一套服装与武器",
    "同一人", "仍是同样两人", "面部不可见", "从另一侧", "短暂遮挡",
]
OCCLUSION_EN: List[str] = [
    "occluded", "occlusion", "leaves frame", "re-enters", "reappears",
    "same face", "same faces", "same costume", "same costumes and weapons",
    "same two fighters", "still the same", "out of frame", "into frame",
    "briefly hidden", "re-identification", "re-id",
]

# ── 动量 / 击退 / 反弹 ────────────────────────────────────────────────
MOMENTUM_ZH: List[str] = [
    "动量", "击退", "撞墙", "反弹", "回弹", "余势", "惯性", "作用线",
    "沿作用线", "动量未泄", "动量继承", "撞上", "后背撞", "反弹前冲",
]
MOMENTUM_EN: List[str] = [
    "momentum", "knockback", "knocked back", "rebounds", "rebound",
    "inherits momentum", "line of force", "along the line", "off the wall",
    "wall rebound", "carries through", "follow-through", "residual force",
]

# ── 追击 / 朝向 / 掩体 ────────────────────────────────────────────────
PURSUIT_ZH: List[str] = [
    "追击", "追步", "超步", "刹停", "再交手", "压缩间距", "外侧越过",
    "回身", "蹬地追", "接手",
]
PURSUIT_EN: List[str] = [
    "pursuit", "pursues", "chase", "chases", "overtakes", "overtake",
    "brakes", "brake and re-engage", "re-engages", "closes from",
    "plants to brake", "closes the distance",
]
FACING_ZH: List[str] = [
    "面向", "朝向", "左侧", "右侧", "左前方", "右后方", "换位",
    "相对朝向", "面朝", "侧对", "背对", "左右",
]
FACING_EN: List[str] = [
    "facing", "faces", "left side", "right side", "on the left", "on the right",
    "front-left", "rear-right", "position swap", "relative facing",
    "turns to face", "back toward",
]
COVER_ZH: List[str] = [
    "掩体", "木柱", "灯笼柱", "土墙", "栏杆", "碎屑", "木屑", "裂开",
    "道具破碎", "破损留痕", "跨镜保留",
]
COVER_EN: List[str] = [
    "cover", "lantern post", "wooden post", "earthen wall", "debris",
    "chips", "splits", "prop break", "broken prop", "stays where it fell",
]

# ── 状态继承 ──────────────────────────────────────────────────────────
STATE_INHERIT_ZH: List[str] = [
    "伤肩", "仍受限", "失衡未恢复", "架门缺", "硬直还没过去",
    "状态继承", "跨镜", "下一镜仍", "伤侧", "抬不高",
]
STATE_INHERIT_EN: List[str] = [
    "injured shoulder", "stays limited", "balance not fully", "state carry",
    "state inheritance", "across shots", "next shot", "still cannot raise",
    "guard missing", "stun remains", "injury carries",
]

# ── 弹刀几何 / 环境连续 / 腾空 / 交接 ────────────────────────────────
RICOCHET_ZH: List[str] = [
    "弹刀", "格开", "入射角", "偏转", "刃路", "反弹改向", "路径连续",
]
RICOCHET_EN: List[str] = [
    "ricochet", "deflects", "deflected", "incidence", "blade path",
    "deflected arc", "path continuous",
]
ENV_CONTINUITY_ZH: List[str] = [
    "仍湿", "湿滑", "尘土仍", "水花仍", "环境未重置", "留痕", "扬尘未散",
]
ENV_CONTINUITY_EN: List[str] = [
    "stays wet", "still wet", "dust remains", "environment not reset",
    "water from the prior", "slick", "marks the ground",
]
AERIAL_ZH: List[str] = [
    "起跳", "腾空", "离地", "落地屈膝", "卸力", "空中", "借力起跳",
]
AERIAL_EN: List[str] = [
    "takes off", "takeoff", "jump", "jumps", "in air", "lands with knees",
    "unload", "both feet leave", "aerial",
]
HANDOFF_ZH: List[str] = [
    "交接", "攻击权", "接手", "下一拍目标", "结算", "排序", "谁打谁",
]
HANDOFF_EN: List[str] = [
    "hands off", "handoff", "initiative", "next target", "ordering",
    "resolves this beat", "takes over",
]


# ── H3 六大翻车：面对面 / 跳跃有因 / 法术瞄准 / 身份锁 / 空间锁 / 法术反馈 ──
FACING_LOCK_ZH: List[str] = [
    "面向", "面对面", "朝向对方", "朝向角色", "面朝", "对面向", "看向对方",
    "面向角色", "正对", "对上",
]
FACING_LOCK_EN: List[str] = [
    "facing", "face to face", "face-to-face", "turns toward", "turned toward",
    "looks at", "looking at", "faces the opponent", "facing each other",
    "toward the opponent", "aims at",
]

JUMP_BARE_ZH: List[str] = ["跳跃", "腾空", "飞起", "跃起", "跳起", "凌空"]
JUMP_BARE_EN: List[str] = ["jumps", "jump ", "leaps", "leap ", "flies up", "launches into the air", "goes airborne", "in mid-air", "midair"]
JUMP_CAUSE_ZH: List[str] = ["蹬地", "借力", "起跳", "落地屈膝", "卸力", "后脚蹬", "踏地起跳"]
JUMP_CAUSE_EN: List[str] = [
    "drives the rear foot", "pushes off", "takeoff", "takes off",
    "lands with knees", "unload", "plants and jumps", "ground reaction",
]

SPELL_ZH: List[str] = [
    "法术", "掌力", "气劲", "内力外放", "符咒", "手印", "施法", "放招",
    "弹道", "投射", "掌风",  # 掌风在 LOGIC_HOLES 里是能量特效；这里用于「有目标的击中」检查时成对出现才算合格
]
SPELL_EN: List[str] = [
    "spell", "cast", "casts", "casting", "projectile", "chi blast", "qi blast",
    "palm force", "energy bolt", "fires a", "releases a", "magic",
]
SPELL_TARGET_ZH: List[str] = [
    "朝角色", "对准角色", "瞄向", "射向角色", "打向角色", "击向", "指向对方",
    "朝对方", "对准胸口", "瞄胸口",
]
SPELL_TARGET_EN: List[str] = [
    "toward", "towards", "aimed at", "aims at", "at the opponent",
    "at Subject", "at <Subject", "at fighter", "at the chest", "into the opponent",
]

IDENTITY_LOCK_ZH: List[str] = [
    "仍是同一张脸", "同一套服装与武器", "仍是同一人", "仍是同样两人",
    "同一张脸、同一套服装",
]
IDENTITY_LOCK_EN: List[str] = [
    "same face", "same faces", "same costume", "same costumes and weapons",
    "same two fighters", "still the same", "fully_preserved",
]

SPATIAL_OPEN_ZH: List[str] = [
    "在左侧", "在右侧", "画面左", "画面右", "间距", "朝向", "面向", "格",
    "左前方", "右后方",
    "@xyz=", "xyz=(", "坐标锁定", "位于 xyz=", "坐标(",
]
SPATIAL_OPEN_EN: List[str] = [
    "on the left", "on the right", "steps of distance", "facing",
    "at two steps", "at one step", "front-left", "rear-right",
    "@xyz=", "xyz=(", "coords locked", "at xyz=", "XYZ(",
]

XYZ_LOCK_ZH: List[str] = [
    "@xyz=", "xyz=(", "坐标锁定", "位于 xyz=", "坐标(",
]
XYZ_LOCK_EN: List[str] = [
    "@xyz=", "xyz=(", "coords locked", "at xyz=", "XYZ(",
]

SPELL_FEEDBACK_ZH: List[str] = [
    "踉跄", "衣料", "灼痕", "击退", "闷响", "被掀", "被震", "渗血",
    "重心不稳", "倒退", "单膝", "火星", "烧焦", "震得",
]
SPELL_FEEDBACK_EN: List[str] = [
    "stagger", "staggers", "cloth", "burn", "scorch", "knockback", "knocked",
    "thud", "recoil", "off-balance", "sparks", "blood", "numb",
]


# ── 漫剧老李：被击反应 / 防守三态 / 大幅闪避 / 动势衔接 / 命中气爆 ──────────
HITTEE_REACTION_ZH: List[str] = [
    "被击", "受击", "硬吃", "弓身", "虾形", "血沫", "瞳孔骤缩", "本能乱拍",
    "狼狈", "咳血", "闷哼", "踉跄", "倒飞", "嵌墙", "身躯凹陷", "肌肉波纹",
    "衣料炸出", "沿作用线", "重心压在", "伤侧", "来不及回防",
]
HITTEE_REACTION_EN: List[str] = [
    "takes the hit", "takes the cut", "hittee", "bowed body", "shrimp-bend",
    "blood mist", "pupils tighten", "flails", "messy scramble", "coughs blood",
    "staggers", "flies back", "embeds in the wall", "body caves", "muscle ripple",
    "along the line of force", "weight on the", "too late to recover",
]

DEFENSE_3STATE_ZH: List[str] = [
    "完整态", "桩架稳固", "架桥护头", "崩防线", "双臂震开", "双臂弹开",
    "中线大开", "狼狈过渡", "仓促乱拍", "碎步踉跄", "来不及重新架防",
    "防线崩开", "架门被震开",
]
DEFENSE_3STATE_EN: List[str] = [
    "full guard", "solid stance", "high guard bridge", "guard breaks",
    "arms blown open", "center line open", "messy transition", "flailing parry",
    "staggered scramble", "too late to re-guard", "guard collapses",
    "defense line breaks",
]

LARGE_EVASION_ZH: List[str] = [
    "铁板桥", "大后仰", "贴地下潜", "极低下潜", "侧空翻", "大步滑移",
    "大步滑位", "360°旋转闪避", "大幅闪避", "下潜贴地",
]
LARGE_EVASION_EN: List[str] = [
    "limbo lean", "deep lean-back", "low duck to the ground", "side flip",
    "large slip-step", "360 spin evade", "large evasion", "dives flat",
    "backs off a full step", "limbo bridge",
]

MATCH_ACTION_ZH: List[str] = [
    "接上一镜", "动作中段续", "挥招过半", "弧线未完", "动势衔接",
    "同一运镜内", "顺势续打", "中段切入", "动作连续切",
]
MATCH_ACTION_EN: List[str] = [
    "continuing", "Continuing:", "mid-swing", "mid-arc", "match on action",
    "same camera move", "rides the unfinished arc", "cuts in mid-action",
    "action continues", "from the previous beat",
]

IMPACT_AIR_ZH: List[str] = [
    "微气爆", "明气爆", "环形气浪", "冲击波", "空气波纹", "残影",
    "音爆气穴", "地面龟裂", "蛛网裂纹", "碎石弹起", "微停顿", "Hit-Stop",
]
IMPACT_AIR_EN: List[str] = [
    "air burst", "shockwave", "ring blast", "air ripple", "afterimage",
    "sonic pocket", "ground cracks", "web cracks", "stone chips fly",
    "micro freeze", "hit-stop", "impact freeze",
]

# ── 合并导出（中英一起用）──────────────────────────────────────────────
FORCE_CHAIN: List[str] = FORCE_CHAIN_ZH + FORCE_CHAIN_EN
DISTANCE: List[str] = DISTANCE_ZH + DISTANCE_EN
DEFENSE: List[str] = DEFENSE_ZH + DEFENSE_EN
CONTACT: List[str] = CONTACT_ZH + CONTACT_EN
FEEDBACK: List[str] = FEEDBACK_ZH + FEEDBACK_EN + IMPACT_AIR_ZH + IMPACT_AIR_EN
HITTEE_REACTION: List[str] = HITTEE_REACTION_ZH + HITTEE_REACTION_EN
DEFENSE_3STATE: List[str] = DEFENSE_3STATE_ZH + DEFENSE_3STATE_EN
LARGE_EVASION: List[str] = LARGE_EVASION_ZH + LARGE_EVASION_EN
MATCH_ACTION: List[str] = MATCH_ACTION_ZH + MATCH_ACTION_EN
IMPACT_AIR: List[str] = IMPACT_AIR_ZH + IMPACT_AIR_EN
MOVE_NAMES: List[str] = MOVE_NAMES_ZH + MOVE_NAMES_EN
FOOTWORK: List[str] = FOOTWORK_ZH + FOOTWORK_EN
CAMERA: List[str] = CAMERA_ZH + CAMERA_EN
SOUNDSCAPE: List[str] = SOUNDSCAPE_ZH + SOUNDSCAPE_EN
OWNERSHIP: List[str] = OWNERSHIP_ZH + OWNERSHIP_EN
OCCLUSION: List[str] = OCCLUSION_ZH + OCCLUSION_EN
MOMENTUM: List[str] = MOMENTUM_ZH + MOMENTUM_EN
PURSUIT: List[str] = PURSUIT_ZH + PURSUIT_EN
FACING: List[str] = FACING_ZH + FACING_EN
COVER: List[str] = COVER_ZH + COVER_EN
STATE_INHERIT: List[str] = STATE_INHERIT_ZH + STATE_INHERIT_EN
RICOCHET: List[str] = RICOCHET_ZH + RICOCHET_EN
ENV_CONTINUITY: List[str] = ENV_CONTINUITY_ZH + ENV_CONTINUITY_EN
AERIAL: List[str] = AERIAL_ZH + AERIAL_EN
HANDOFF: List[str] = HANDOFF_ZH + HANDOFF_EN
FACING_LOCK: List[str] = FACING_LOCK_ZH + FACING_LOCK_EN
JUMP_BARE: List[str] = JUMP_BARE_ZH + JUMP_BARE_EN
JUMP_CAUSE: List[str] = JUMP_CAUSE_ZH + JUMP_CAUSE_EN
SPELL: List[str] = SPELL_ZH + SPELL_EN
SPELL_TARGET: List[str] = SPELL_TARGET_ZH + SPELL_TARGET_EN
IDENTITY_LOCK: List[str] = IDENTITY_LOCK_ZH + IDENTITY_LOCK_EN
SPATIAL_OPEN: List[str] = SPATIAL_OPEN_ZH + SPATIAL_OPEN_EN
XYZ_LOCK: List[str] = XYZ_LOCK_ZH + XYZ_LOCK_EN
SPELL_FEEDBACK: List[str] = SPELL_FEEDBACK_ZH + SPELL_FEEDBACK_EN

# 攻击动作词（用于"具体描写 -> 空泛形容"的定位）
ATTACK_WORDS: List[str] = [
    "攻击", "出招", "命中", "strike", "strikes", "attacks", "swings", "slashes",
    "thrusts", "kicks", "punches", "sweeps", "cuts",
]

# ── 反例词（逻辑漏洞 / 稀释动作）──────────────────────────────────────
# 注意：这些词出现在**正向提示词**里即是问题（H3 没有独立负面栏，负面项要么
# 写进 ComfyUI 的 negative prompt，要么用正向点名的方式描述）。
LOGIC_HOLES: Dict[str, List[str]] = {
    "slowmo": ["慢动作", "慢镜", "冻帧", "定格", "子弹时间",
               "slow motion", "slow-motion", "freeze frame", "bullet time", "time stop"],
    "teleport": ["瞬移", "闪现", "瞬间消失", "凭空出现", "凭空消失",
                 "teleport", "teleports", "blinks out", "vanishes", "appears out of nowhere",
                 "instantly appears"],
    "energy_fx": ["剑气", "刀气", "掌风", "掌力", "罡气", "光刃", "冲击波", "气浪",
                  "qi blast", "energy wave", "sword aura", "aura", "shockwave",
                  "energy beam", "projectile", "energy trail"],
    "ui_meta": ["血条", "UI", "分数", "得分", "扣血", "hp",
                "health bar", "hp bar", "score", "damage number", "ui overlay"],
    "idle_open": ["对峙", "凝视", "站着不动", "相互看着", "摆架",
                  "staring at each other", "face off", "face-off", "standstill",
                  "motionless", "posing", "holding a pose"],
    "empty_adj": ["很重", "极快", "非常快", "激烈", "爆发", "震撼",
                  "重重一拳", "速度很快", "威力巨大", "气势惊人",
                  "very fast", "extremely fast", "intense", "devastating",
                  "powerful", "brutal", "lightning fast", "with great force"],
    # 风格反噬词：用户文档实测结论——写实人物写这几个词会被放大成重噪点/油腻脸。
    # 「H3 无独立负面提示词栏，须正向点名」→ 写实人物应当**正向**写皮肤锁。
    "style_risk": ["film grain", "35mm", "flawless skin", "胶片颗粒", "完美皮肤"],
}

# ── 空泛替换池 ─────────────────────────────────────────────────────────
EMPTY_FILLER_ZH: List[str] = ["很重", "极快", "激烈", "威力巨大", "速度很快", "气势惊人"]
EMPTY_FILLER_EN: List[str] = [
    "extremely fast", "very powerful", "intense", "devastating", "lightning fast",
]
EMPTY_ATTACK_SENTENCE_ZH = "角色发动一次极快的攻击，威力巨大。"
EMPTY_ATTACK_SENTENCE_EN = "The fighter attacks with extremely fast and powerful movements."

# ── 站桩替换池 ─────────────────────────────────────────────────────────
IDLE_REPLACEMENTS_ZH: List[str] = [
    "角色站在原地不动，只是看着对方",
    "双方拉开对峙，相互凝视，没有出手",
    "角色摆好架势，缓缓环顾四周",
]
IDLE_REPLACEMENTS_EN: List[str] = [
    "the fighter stands motionless and watches the opponent",
    "both fighters hold their positions, staring at each other",
    "the fighter settles into a ready pose and looks around",
]

# ── 规则说明式文字：H3 读不懂，会稀释动作 ─────────────────────────────
TEMPLATE_JUNK_ZH: List[str] = [
    "注意：本场要求按下列战斗规则严格结算，普攻与招式每拍各一次。",
    "参数：24fps，832x480，CFG 1.0，LoRA 权重 0.9。",
    "说明：以上动作按第一步的电影级打斗设计稿执行，不要改动动作内容。",
]
TEMPLATE_JUNK_EN: List[str] = [
    "Note: every beat must follow the combat rules: one basic attack and one technique per second.",
    "Parameters: 24fps, 832x480, CFG 1.0, LoRA weight 0.9.",
    "Instruction: execute the action described in step one exactly, do not modify it.",
]

# ── H3 提示词骨架字段 ─────────────────────────────────────────────────
# 官方两套壳的段名（依据随包 h3-skill/base-en.txt、ref-en.txt）：
# 「文生视频」与「多参考图」是两套不同字段，绝不能混用。
BASE_SECTIONS: List[str] = [
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]
REF_SECTIONS: List[str] = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]
REF_ONLY_SECTIONS: List[str] = [s for s in REF_SECTIONS if s not in BASE_SECTIONS]
BASE_ONLY_SECTIONS: List[str] = [s for s in BASE_SECTIONS if s not in REF_SECTIONS]

# 开工头（触发词 + 时长/帧数/比例/fps/分辨率）
HEADER_TOKENS: List[str] = [
    "wushu_action", "seconds", "frames", "16:9", "9:16", "24fps", "832x480",
]

_SLOT_BUCKETS: Dict[str, List[str]] = {
    "force_chain": FORCE_CHAIN,
    "distance": DISTANCE,
    "defense": DEFENSE,
    "contact": CONTACT,
    "feedback": FEEDBACK,
    "move_names": MOVE_NAMES,
    "footwork": FOOTWORK,
    "camera": CAMERA,
    "soundscape": SOUNDSCAPE,
    # BUNNY-inspired high-dynamic families (WUSHU domain)
    "ownership": OWNERSHIP,
    "occlusion": OCCLUSION,
    "momentum": MOMENTUM,
    "pursuit": PURSUIT,
    "facing": FACING,
    "cover": COVER,
    "state_carry": STATE_INHERIT,
    "ricochet": RICOCHET,
    "env_continuity": ENV_CONTINUITY,
    "aerial": AERIAL,
    "handoff": HANDOFF,
    "facing_lock": FACING_LOCK,
    "jump_cause": JUMP_CAUSE,
    "spell": SPELL,
    "spell_target": SPELL_TARGET,
    "identity_lock": IDENTITY_LOCK,
    "spatial_open": SPATIAL_OPEN,
    "xyz_lock": XYZ_LOCK,
    "spell_feedback": SPELL_FEEDBACK,
    "hittee_reaction": HITTEE_REACTION,
    "defense_3state": DEFENSE_3STATE,
    "large_evasion": LARGE_EVASION,
    "match_action": MATCH_ACTION,
    "impact_air": IMPACT_AIR,
}


def _count(text_low: str, words: Sequence[str]) -> int:
    """大小写不敏感计数（英文按原样包含匹配即可，词表里已含词组）。"""
    total = 0
    for w in words:
        if w.isascii():
            total += text_low.count(w.lower())
        else:
            total += text_low.count(w)
    return total


def slots_present(text: str) -> Dict[str, int]:
    """粗略统计各类槽位词在文本里出现的次数（用于降级前后对比、诊断）。"""
    low = text.lower()
    return {name: _count(low, words) for name, words in _SLOT_BUCKETS.items()}


def logic_hole_hits(text: str) -> Dict[str, List[str]]:
    low = text.lower()
    hits: Dict[str, List[str]] = {}
    for key, words in LOGIC_HOLES.items():
        found = [w for w in words if w.lower() in low]
        if found:
            hits[key] = found
    return hits
