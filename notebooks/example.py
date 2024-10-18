# %%
from cbspython import domain_feature_list
from IPython.display import display

import brainbodydata as bbd

# %%
# Let's suppress Datalad output for this, because it generates a lot of detail.
dl_display = "disabled"

paaq = bbd.PAAQ(dl_renderer=dl_display)
vgq = bbd.VGQ(dl_renderer=dl_display)
cogs = bbd.BBCogsData(dl_renderer=dl_display)
demo = bbd.BBQuestionnaire(dl_renderer=dl_display)
ml = bbd.BBMasterList.from_frozen()

# %%
# Check out the demogrpahics questionnaire data
display(demo.data.info())
demo.data.tail()

# %%
# Check out the VGQ data
display(vgq.data.info())
vgq.data.tail()

# %%
# Check out the PAAQ data
display(paaq.data.info())
paaq.data.tail()

# %%
# And finally the cog scores
display(cogs.scores.info(max_cols=1000))
cogs.scores.tail()

# %%
# Might as well take a look at the primary test scores, too.
cogs.scores[domain_feature_list(abbreviated=True)].describe().T

# %%
users = bbd.BBMasterList.from_frozen()
display(users.data)
