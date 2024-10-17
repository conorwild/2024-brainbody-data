# %%
import brainbodydata as bbs

# %%
old = bbs.BBMasterList.from_frozen().data
new = bbs.BBMasterList()

if old.shape[0] < new.data.shape[0]:
    new.freeze(
        datalad_commit=True,
        commit_message="SCI99 updating master list",
    )
# %%
