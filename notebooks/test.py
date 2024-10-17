# %%
from IPython.display import display
import brainbodydata as bbd

# %%
paaq = bbd.PAAQ(dl_renderer="disabled")

# %%
vgq = bbd.VGQ(dl_renderer="disabled")
cogs = bbd.BBCogsData(dl_renderer="disabled")
demo = bbd.BBQuestionnaire(dl_renderer="disabled")
ml = bbd.BBMasterList.from_frozen()
# %%
# Check out the demogrpahics questionnaire data
display(demo.data.info())
demo.data.tail()

# %%
# Check out the VGQ data
vgq = bbd.VGQ(dl_renderer="disabled")
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
users = bbd.BBMasterList.from_frozen()
display(users.data)


# %%
