from .cornernavenv import CornerNavEnv
from .lightdarkenv import LightDarkEnv
from .intentiontagenv import IntentionTagEnv
from .puckpushenv import PuckPushEnv
from gym.envs.registration import register

register(
    id='CornerNav-v0',
    entry_point='gymenvs.cornernavenv:CornerNavEnv',
)

register(
    id='LightDark-v0',
    entry_point='gymenvs.lightdarkenv:LightDarkEnv',
)

register(
    id='IntentionTag-v0',
    entry_point='gymenvs.intentiontagenv:IntentionTagEnv',
)

register(
    id='PuckPush-v0',
    entry_point='gymenvs.puckpushenv:PuckPushEnv',
)
