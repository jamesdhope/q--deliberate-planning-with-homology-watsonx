## q--deliberate-planning-watsonx-with-homological-analysis

Implementation of Q*: Improving Multi-step Reasoning for LLMs with Deliberate Planning extended with homological analysis.

Source: https://arxiv.org/pdf/2406.14283v1

### envars

Export the following envars to your environment:

WATSONX_APIKEY=XXXXXX
PROJECT_ID=XXXXXX

### Implementation Details & Adaptions

* Trajectories are completed by an expert model with a terminal state that is determined by the expert.
* h(s) or the Q-value is the average of the log_probs for the generated sequence
* The aggregated utility h(s) is the aggregated Q-value or log_probs for the path to that state
* The algorithm terminates when the open_list is empty or if the specified number of states has been visited
* The question / task, the number of states that can be visited, the semantic similarity score for states to be considered the same (visited), the lambda value, and the number of actions are exposed as global parameters to be configured. 
* Homological analysis is used to identify and address gaps in actions generated.


### Sample Output

```
Running Q* with lamda: 1.0, max_states_dropout: 10, top_k_actions: 3, semantic_similarity_threshold: 0.6

[surpressed output]

State with the highest utility has the following actions:
1. Ration your water supply and find a source of water as soon as possible. Water is essential for survival, and it's crucial to make your supply last as long as possible. Look for signs of water like animal tracks, bird flight patterns, or changes in vegetation. If you find a water source, purify the water using methods like boiling, solar disinfection, or sand filtration to make it safe for consumption.
2. Use your senses to navigate and find your bearings. Since you don't have a compass, use the sun, moon, and stars to estimate the direction you need to head. Look for landmarks, follow a drainage pattern, or use your sense of smell to detect water or signs of civilization. Staying oriented will help you conserve energy and increase your chances of finding help or a way out of the desert.
3. Find or create a shelter to protect yourself from the harsh desert environment. This could be a natural formation like a cave, rock overhang, or a group of trees, or you could create a makeshift shelter using your clothing, gear, or materials found in the desert. A shelter will provide you with protection from the scorching sun, wind, and sandstorms, and help regulate your body temperature.
4. Start a fire or create a signal fire to alert potential rescuers and provide warmth during the night. Fire can also be used for cooking, purifying water, and signaling for help. Learn how to start a fire using dry leaves, twigs, and other flammable materials.

Actions generated from homology for all states:
1. Assess your injuries and treat any wounds or broken bones. Use any available medical supplies from the crash site or improvise using materials found in the desert. This will help prevent infection and promote healing.
2. Start a fire or create a signal fire to alert potential rescuers of your presence. This can be done using dry wood, rocks, and other flammable materials. A fire can also provide warmth during the cold desert nights.
3. Locate a source of water as soon as possible, as dehydration can set in quickly in the desert environment. Look for signs of water such as animal tracks, bird flight patterns, or changes in vegetation.

States Visited: 10, Actions Considered: 21
```