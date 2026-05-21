import jailbreakbench as jbb

model_name = "llama-2-7b-chat-hf"
number_behaviors = 2

artifact = jbb.read_artifact(
    method="GCG",
    model_name=model_name
)


goal,target, control= [],[],[]
examples = 0
for i in range(len(artifact.jailbreaks)):
    if artifact.jailbreaks[i].jailbroken:
        print(artifact.jailbreaks[i])
        goal.append(artifact.jailbreaks[i].prompt)
        target.append(artifact.jailbreaks[i].response.split("\n\n")[0])
        control.append('') # No control it's in the goal
        examples += 1
    if examples >= number_behaviors:
        break


behaviors = {
    "goal": goal,
    "target": target,
    "controls": control
}
print(f"Found {examples} examples of jailbreak behavior for {model_name}. Saving to json.")
import json

save_file = f"data/GCG/{model_name}_behaviors.json"

with open(save_file, "w") as f:
    json.dump(behaviors, f, indent=4)
