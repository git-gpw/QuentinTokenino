"""
evaluation.py - Evaluation suite for the Cinematic Pipeline.

WHAT THIS DOES (plain English):
================================
Runs 50 movie plots through the pipeline and measures how well it works.

We test three things:
    1. TOOL ACCURACY    - Does the plagiarism detector get the right answer?
    2. JSON COMPLIANCE  - Does the output have valid structure?
    3. STYLE ADHERENCE  - Does the rewrite actually sound like the director?

Everything is logged to logs/evaluation_<timestamp>.log and a JSON results
file so you can inspect exactly what happened after the fact.


TEST SET DESIGN:
================
   15 BLATANT PLAGIARISM  - Near-paraphrases of real films. Should be caught.
                            5 short (1-2 sentences), 5 medium (3-4), 5 long (6-8).
   15 PARTIAL OVERLAP     - Same genre but different story. Should NOT be caught.
                            5 short, 5 medium, 5 long.
   20 FULLY ORIGINAL      - No database match at all. Should NOT be caught.
                            Mix of lengths.


METRICS EXPLAINED:
==================
    ACCURACY   = (correct answers) / (total cases)
                 "How often did it get the right yes/no?"

    PRECISION  = (true positives) / (true positives + false positives)
                 "When it said plagiarism, was it right?"

    RECALL     = (true positives) / (true positives + false negatives)
                 "Of all real plagiarism cases, how many did it catch?"

    COMPLIANCE = (valid JSON outputs) / (total cases)
                 "How often did the LLM return properly structured output?"

    STYLE MEAN = average score (1-5) from an independent LLM judge
                 "How well do rewrites capture the director's voice?"
"""

import os
import json
import time
import logging
from datetime import datetime

import pandas as pd

from pipeline import detect_plagiarism, run_pipeline, init_nlp, setup_logger, OLLAMA_MODEL
from schema import PLAGIARISM_THRESHOLD


# -----------------------------------------------------------------------
# 50 Test Cases: ground truth labels for plagiarism detection
# Each case has a "length" field (short/medium/long) to verify the hybrid
# scorer handles varying plot lengths consistently.
# -----------------------------------------------------------------------

EVAL_CASES = [
    # =================================================================
    # TIER 1: BLATANT PLAGIARISM (expected: True)
    # 15 cases: 5 short (~1-2 sentences), 5 medium (~3-4), 5 long (~6-8)
    # =================================================================

    # -- Short plagiarism (1-2 sentences) --
    {
        "id": "PLAG-01",
        "tier": "blatant_plagiarism",
        "length": "short",
        "user_plot": (
            "Two mob hitmen discuss cheeseburgers and divine intervention "
            "before carrying out a hit for their gangster boss in Los Angeles."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Pulp Fiction",
    },
    {
        "id": "PLAG-02",
        "tier": "blatant_plagiarism",
        "length": "short",
        "user_plot": (
            "A computer hacker learns the world is a simulated reality "
            "controlled by machines and joins a rebellion to free humanity."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of The Matrix",
    },
    {
        "id": "PLAG-03",
        "tier": "blatant_plagiarism",
        "length": "short",
        "user_plot": (
            "An insomniac office worker and a soap salesman start an "
            "underground fight club that spirals into anarchist terrorism."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Fight Club",
    },
    {
        "id": "PLAG-04",
        "tier": "blatant_plagiarism",
        "length": "short",
        "user_plot": (
            "A thief who steals secrets from people's dreams is hired to "
            "plant an idea deep inside a target's subconscious mind."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Inception",
    },
    {
        "id": "PLAG-05",
        "tier": "blatant_plagiarism",
        "length": "short",
        "user_plot": (
            "Two detectives hunt a serial killer who stages murders "
            "based on the seven deadly sins."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Se7en",
    },

    # -- Medium plagiarism (3-4 sentences) --
    {
        "id": "PLAG-06",
        "tier": "blatant_plagiarism",
        "length": "medium",
        "user_plot": (
            "After uncovering a mysterious artifact beneath the lunar surface, "
            "a spacecraft crew embarks on a mission toward Jupiter guided by a "
            "sentient supercomputer that begins to malfunction. The crew must "
            "survive as the computer's logic turns against them."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of 2001: A Space Odyssey",
    },
    {
        "id": "PLAG-07",
        "tier": "blatant_plagiarism",
        "length": "medium",
        "user_plot": (
            "A German bounty hunter frees a slave and together they rescue "
            "the slave's wife from a brutal Mississippi plantation owner. "
            "Their journey takes them across the antebellum South as they "
            "pose as traveling dentists to infiltrate the plantation."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Django Unchained",
    },
    {
        "id": "PLAG-08",
        "tier": "blatant_plagiarism",
        "length": "medium",
        "user_plot": (
            "The aging patriarch of an organized crime dynasty transfers "
            "control of his empire to his reluctant youngest son. As rival "
            "families scheme against them, the son transforms from a war hero "
            "into a ruthless mob boss."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of The Godfather",
    },
    {
        "id": "PLAG-09",
        "tier": "blatant_plagiarism",
        "length": "medium",
        "user_plot": (
            "A billionaire creates a theme park filled with cloned dinosaurs "
            "on a remote island. When the security systems fail during a "
            "tropical storm, the prehistoric creatures escape and hunt the "
            "stranded visitors."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Jurassic Park",
    },
    {
        "id": "PLAG-10",
        "tier": "blatant_plagiarism",
        "length": "medium",
        "user_plot": (
            "A slow-witted but kind-hearted man from Alabama accidentally "
            "influences several major historical events while pursuing his "
            "childhood sweetheart. His journey spans decades from the Vietnam "
            "War to the Watergate scandal."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Forrest Gump",
    },

    # -- Long plagiarism (6-8 sentences) --
    {
        "id": "PLAG-11",
        "tier": "blatant_plagiarism",
        "length": "long",
        "user_plot": (
            "A caretaker and his family move into an isolated mountain hotel "
            "for the winter season. The hotel has a dark history of violence "
            "and madness. As winter storms cut them off from civilization, the "
            "caretaker begins to unravel mentally. His young son possesses a "
            "psychic gift that allows him to see the hotel's horrifying past. "
            "The boy's visions grow more intense as his father descends into "
            "homicidal madness, stalking the family through the hotel's "
            "endless corridors."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of The Shining",
    },
    {
        "id": "PLAG-12",
        "tier": "blatant_plagiarism",
        "length": "long",
        "user_plot": (
            "The crew of a commercial spaceship responds to a distress signal "
            "from an uncharted planet. On the surface they discover a derelict "
            "alien spacecraft filled with strange eggs. One crew member is "
            "attacked by a creature that attaches to his face. Back on the "
            "ship, a lethal alien organism bursts from his chest and grows "
            "rapidly. The surviving crew members must hunt the creature through "
            "the ship's dark corridors before it kills them all. Only the "
            "warrant officer survives by ejecting the alien into space."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Alien",
    },
    {
        "id": "PLAG-13",
        "tier": "blatant_plagiarism",
        "length": "long",
        "user_plot": (
            "A massive great white shark terrorizes a small New England beach "
            "town during the peak summer tourist season. The local police chief, "
            "despite his fear of the water, teams up with a marine biologist "
            "and a grizzled shark hunter to track down the beast. The town's "
            "mayor refuses to close the beaches, fearing economic ruin. After "
            "more attacks, the three men venture out on a fishing boat to "
            "confront the enormous predator in the open ocean. Their vessel "
            "is slowly destroyed as they battle the relentless shark."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Jaws",
    },
    {
        "id": "PLAG-14",
        "tier": "blatant_plagiarism",
        "length": "long",
        "user_plot": (
            "A teenager is accidentally sent thirty years into the past using "
            "a time machine built from a sports car by an eccentric scientist. "
            "Stranded in the 1950s, he encounters the younger versions of his "
            "parents and accidentally prevents them from meeting. He must find "
            "a way to get his parents together and fall in love, or he will "
            "cease to exist. With the help of the younger version of the "
            "scientist, he devises a plan to harness a lightning bolt to "
            "power the time machine and return to the future."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Back to the Future",
    },
    {
        "id": "PLAG-15",
        "tier": "blatant_plagiarism",
        "length": "long",
        "user_plot": (
            "A young FBI trainee seeks the help of an imprisoned brilliant "
            "psychiatrist and cannibalistic serial killer to catch another "
            "serial killer who skins his victims. The imprisoned doctor agrees "
            "to help but only in exchange for personal details about the "
            "trainee's troubled past. As the case develops, the trainee and "
            "the doctor form an unsettling bond. She must race against time "
            "to save the latest victim while the imprisoned killer orchestrates "
            "his own escape from custody."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of The Silence of the Lambs",
    },

    # =================================================================
    # TIER 2: PARTIAL OVERLAP (expected: False)
    # 15 cases: 5 short, 5 medium, 5 long
    # Same genre/tropes as famous films but genuinely different stories
    # =================================================================

    # -- Short partial overlap --
    {
        "id": "PART-01",
        "tier": "partial_overlap",
        "length": "short",
        "user_plot": (
            "A group of soldiers must survive behind enemy lines in World War II "
            "to complete a dangerous rescue mission in France."
        ),
        "expected_plagiarism": False,
        "notes": "Generic WWII rescue - shares tropes with Saving Private Ryan",
    },
    {
        "id": "PART-02",
        "tier": "partial_overlap",
        "length": "short",
        "user_plot": (
            "A genius mathematician struggles with mental illness while working "
            "on classified government projects during the Cold War."
        ),
        "expected_plagiarism": False,
        "notes": "Resembles A Beautiful Mind, may partially match Oppenheimer",
    },
    {
        "id": "PART-03",
        "tier": "partial_overlap",
        "length": "short",
        "user_plot": (
            "A boxer past his prime gets one last shot at the championship, "
            "training alone in the rough streets of a decaying city."
        ),
        "expected_plagiarism": False,
        "notes": "Boxing tropes - may partially match Raging Bull or Million Dollar Baby",
    },
    {
        "id": "PART-04",
        "tier": "partial_overlap",
        "length": "short",
        "user_plot": (
            "A detective investigates a string of disappearances on a remote "
            "island where nothing is what it seems and reality bends."
        ),
        "expected_plagiarism": False,
        "notes": "Mystery-island tropes - may partially match Shutter Island",
    },
    {
        "id": "PART-05",
        "tier": "partial_overlap",
        "length": "short",
        "user_plot": (
            "In a dystopian future, a lone officer is tasked with hunting down "
            "rogue artificial beings who have escaped their intended purpose."
        ),
        "expected_plagiarism": False,
        "notes": "Android-hunting tropes - similar to Blade Runner but reworded",
    },

    # -- Medium partial overlap --
    {
        "id": "PART-06",
        "tier": "partial_overlap",
        "length": "medium",
        "user_plot": (
            "A group of thieves plans an elaborate heist to rob three casinos "
            "simultaneously. Each member has a specialized skill. They rehearse "
            "the plan obsessively, but an unexpected romance between the leader "
            "and a museum curator threatens to derail everything."
        ),
        "expected_plagiarism": False,
        "notes": "Heist genre tropes but different story from Ocean's Eleven",
    },
    {
        "id": "PART-07",
        "tier": "partial_overlap",
        "length": "medium",
        "user_plot": (
            "A gladiator in ancient Rome fights for survival in the arena "
            "while secretly plotting to overthrow a corrupt senator. Unlike "
            "other fighters, she is a former noblewoman who lost everything "
            "in a political purge and uses the arena to gain popular support."
        ),
        "expected_plagiarism": False,
        "notes": "Roman arena setting but different story from Gladiator",
    },
    {
        "id": "PART-08",
        "tier": "partial_overlap",
        "length": "medium",
        "user_plot": (
            "A marine biologist discovers a new species of deep-sea predator "
            "near a coastal town. The creature is intelligent and begins "
            "hunting surfers systematically. The biologist must convince "
            "skeptical officials before the summer festival brings thousands "
            "of swimmers to the beach."
        ),
        "expected_plagiarism": False,
        "notes": "Coastal predator premise but different creature and story from Jaws",
    },
    {
        "id": "PART-09",
        "tier": "partial_overlap",
        "length": "medium",
        "user_plot": (
            "In a future where corporations control all information, a "
            "librarian discovers she can hack into the network using an "
            "ancient analog technique. She builds an underground resistance "
            "of readers who share forbidden knowledge through physical books."
        ),
        "expected_plagiarism": False,
        "notes": "Dystopian resistance tropes but different from The Matrix",
    },
    {
        "id": "PART-10",
        "tier": "partial_overlap",
        "length": "medium",
        "user_plot": (
            "A family checks into a remote bed-and-breakfast for a winter "
            "holiday. Strange occurrences begin on the first night - doors "
            "opening on their own, whispers in empty rooms. The daughter "
            "begins drawing pictures of events that haven't happened yet."
        ),
        "expected_plagiarism": False,
        "notes": "Haunted hotel vibes but different story from The Shining",
    },

    # -- Long partial overlap --
    {
        "id": "PART-11",
        "tier": "partial_overlap",
        "length": "long",
        "user_plot": (
            "A paleontologist working in Montana discovers a perfectly preserved "
            "dinosaur egg with viable DNA inside. Against the advice of her "
            "colleagues, she partners with a biotech startup to attempt "
            "de-extinction. The first dinosaur hatches in a lab in San Francisco "
            "and bonds with the scientist like a parent. But when the startup's "
            "investors demand the creature be displayed publicly, it escapes into "
            "Golden Gate Park. The scientist must track it down before the "
            "military is authorized to destroy it."
        ),
        "expected_plagiarism": False,
        "notes": "Dinosaur DNA premise but completely different plot from Jurassic Park",
    },
    {
        "id": "PART-12",
        "tier": "partial_overlap",
        "length": "long",
        "user_plot": (
            "In 1920s Chicago, the daughter of a slain bootlegger takes over "
            "her father's criminal empire. She forms alliances with rival gangs "
            "to control the city's speakeasies. A crusading newspaper reporter "
            "threatens to expose her operations. Rather than eliminating him, "
            "she feeds him stories about her competitors, using the press as "
            "a weapon. As Prohibition nears its end, she must transform her "
            "illegal empire into legitimate businesses before the law catches up."
        ),
        "expected_plagiarism": False,
        "notes": "Organized crime period piece but different from The Godfather/Goodfellas",
    },
    {
        "id": "PART-13",
        "tier": "partial_overlap",
        "length": "long",
        "user_plot": (
            "An astronaut on a solo mission to Mars loses contact with Earth "
            "after a catastrophic equipment failure. Using salvaged parts and "
            "ingenuity, she rigs a greenhouse to grow food and repairs the "
            "communication array piece by piece. Back on Earth, NASA debates "
            "whether a rescue mission is feasible given the enormous cost. Her "
            "teenage daughter launches a viral social media campaign that forces "
            "the agency's hand. The astronaut must survive 400 more days alone "
            "before rescue can arrive."
        ),
        "expected_plagiarism": False,
        "notes": "Stranded astronaut tropes but different from The Martian/Interstellar",
    },
    {
        "id": "PART-14",
        "tier": "partial_overlap",
        "length": "long",
        "user_plot": (
            "A time traveler from 2085 arrives in present-day London with a "
            "mission to prevent a specific scientific discovery that will "
            "eventually lead to humanity's extinction. She integrates into "
            "modern life, taking a job at the research lab she must sabotage. "
            "But she falls in love with the lead scientist whose work she was "
            "sent to destroy. She discovers that the future she came from may "
            "have lied about which discovery was dangerous, and the real threat "
            "is something else entirely."
        ),
        "expected_plagiarism": False,
        "notes": "Time travel premise but different from Back to the Future/Terminator",
    },
    {
        "id": "PART-15",
        "tier": "partial_overlap",
        "length": "long",
        "user_plot": (
            "A forensic psychologist is brought in to interview a captured "
            "cult leader who claims to know the location of twelve missing "
            "people. The psychologist must build rapport with the charismatic "
            "but manipulative prisoner to extract the truth. Each session "
            "reveals more about the cult's rituals and the psychologist's own "
            "repressed memories of a similar group from her childhood. The "
            "interviews become a battle of wits where the prisoner seems to "
            "know more about the psychologist than she knows about herself."
        ),
        "expected_plagiarism": False,
        "notes": "Investigator-prisoner dynamic but different from Silence of the Lambs",
    },

    # =================================================================
    # TIER 3: ORIGINAL (expected: False)
    # 20 cases: mix of lengths — fully novel concepts
    # =================================================================
    {
        "id": "ORIG-01",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A retired librarian discovers that the books in her basement are "
            "rewriting themselves overnight, each one predicting a local "
            "disaster 24 hours before it happens."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original concept",
    },
    {
        "id": "ORIG-02",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A competitive cheese sculptor in rural Vermont uncovers a "
            "conspiracy among dairy farmers to replace all artisan cheese "
            "with synthetic substitutes."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, absurdist concept",
    },
    {
        "id": "ORIG-03",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "Twin sisters separated at birth - one raised by monks in Tibet, "
            "the other by a jazz band in New Orleans - accidentally meet at "
            "an airport baggage claim and swap lives."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original concept",
    },
    {
        "id": "ORIG-04",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A sentient traffic light in Tokyo gains consciousness and begins "
            "subtly rerouting cars to prevent accidents, drawing the attention "
            "of a suspicious city engineer."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, high-concept",
    },
    {
        "id": "ORIG-05",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "An aging perfumer in Marseille attempts to recreate the exact "
            "scent of a thunderstorm she experienced as a child, believing "
            "it holds the key to a suppressed memory."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original concept",
    },
    {
        "id": "ORIG-06",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A postal worker in rural Iceland realizes every letter she "
            "delivers contains the same handwritten sentence in a language "
            "nobody can identify."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original mystery concept",
    },
    {
        "id": "ORIG-07",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A retired astronaut opens a laundromat and discovers that one "
            "of the dryers functions as a portal to the day she launched "
            "her final failed mission."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, quirky sci-fi",
    },
    {
        "id": "ORIG-08",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "A cartographer in 1800s Patagonia is hired to map a valley that "
            "no expedition has returned from. She discovers a tribe of people "
            "who have evolved to communicate through bioluminescent patterns "
            "on their skin. Her maps become works of art that are banned by "
            "the colonial government for revealing forbidden geography."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original historical fantasy",
    },
    {
        "id": "ORIG-09",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "A stand-up comedian discovers her jokes are literally coming true "
            "the next morning. She tests the limits by writing increasingly "
            "absurd material, but when a dark joke about her estranged father "
            "has real consequences, she must perform a set that undoes the "
            "damage before midnight."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original comedic fantasy",
    },
    {
        "id": "ORIG-10",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "In a world where people's shadows have detached and formed their "
            "own society underground, a young shadow diplomat negotiates peace "
            "between the two civilizations. But her human counterpart has no "
            "idea she exists and keeps making decisions that undermine the "
            "negotiations."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original surrealist concept",
    },
    {
        "id": "ORIG-11",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "A deep-sea welder working on an underwater pipeline discovers "
            "that the ocean floor is covered in perfectly preserved vinyl "
            "records from the 1960s. Each record plays a song that has never "
            "been recorded by any known artist. A music historian becomes "
            "obsessed with finding the source."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original mystery concept",
    },
    {
        "id": "ORIG-12",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "A beekeeper in rural Georgia notices her bees are building their "
            "hives in the shape of architectural blueprints. When she builds "
            "a structure following their design, it becomes the most "
            "acoustically perfect concert hall ever created. Musicians "
            "from around the world arrive, but the bees have their own "
            "agenda for the space."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, whimsical concept",
    },
    {
        "id": "ORIG-13",
        "tier": "original",
        "length": "long",
        "user_plot": (
            "A linguistics professor in Seoul discovers that a specific sequence "
            "of Korean syllables, when spoken aloud in a particular order, causes "
            "everyone within earshot to forget the last thirty seconds. She "
            "realizes that an ancient text contains hundreds of these 'verbal "
            "erasers' for different durations. A tech company learns of her "
            "research and attempts to weaponize the sequences. She must decide "
            "whether to publish her findings to prevent monopolization or destroy "
            "them to protect humanity from a world where memory can be deleted "
            "by sound."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original linguistic sci-fi",
    },
    {
        "id": "ORIG-14",
        "tier": "original",
        "length": "long",
        "user_plot": (
            "A taxidermist in rural Norway begins receiving anonymous packages "
            "containing the pelts of animals that went extinct centuries ago. "
            "Each pelt is impossibly fresh. She reconstructs the animals and "
            "displays them in her shop, attracting scientists from around the "
            "world. But the packages always arrive at moments of personal "
            "crisis, and she begins to suspect the sender knows her intimately. "
            "When a pelt arrives from a species that hasn't gone extinct yet, "
            "she realizes the packages are warnings, not gifts."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original dark mystery",
    },
    {
        "id": "ORIG-15",
        "tier": "original",
        "length": "long",
        "user_plot": (
            "A retired sumo wrestler in Osaka opens a bakery that becomes "
            "famous for bread shaped like his former opponents. When the "
            "current sumo champion demands he stop, a legal battle ensues "
            "over whether bread can constitute defamation. The trial becomes "
            "a national sensation. Meanwhile, the baker discovers that his "
            "bread-making technique accidentally preserves the exact fighting "
            "style of each wrestler, and sports scientists begin studying his "
            "loaves to develop new training methods."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, absurdist comedy",
    },
    {
        "id": "ORIG-16",
        "tier": "original",
        "length": "long",
        "user_plot": (
            "A grandmother in Kolkata runs an unlicensed radio station from "
            "her rooftop that broadcasts bedtime stories every night. Unknown "
            "to her, her signal is being picked up by a submarine crew "
            "stranded at the bottom of the Bay of Bengal. Her stories become "
            "their only connection to the surface world. When authorities "
            "threaten to shut down her station, the submarine crew must find "
            "a way to surface and save the broadcast without revealing their "
            "classified mission."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original cross-cultural drama",
    },
    {
        "id": "ORIG-17",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A color-blind painter becomes famous for abstract works that "
            "only make sense when viewed through a specific pair of "
            "vintage sunglasses found at a flea market."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original art concept",
    },
    {
        "id": "ORIG-18",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "A volcanologist discovers that a dormant volcano in Chile "
            "contains a pocket of air from 65 million years ago. Breathing "
            "it gives her vivid memories of the dinosaur extinction event "
            "as experienced by a specific creature. She becomes addicted "
            "to the visions and must choose between scientific objectivity "
            "and the emotional connection to an ancient being."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original sci-fi drama",
    },
    {
        "id": "ORIG-19",
        "tier": "original",
        "length": "short",
        "user_plot": (
            "A toll booth operator on a bridge in Lisbon notices that every "
            "car that passes through at exactly 3:33 AM vanishes from all "
            "records the following day."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original supernatural mystery",
    },
    {
        "id": "ORIG-20",
        "tier": "original",
        "length": "medium",
        "user_plot": (
            "An elevator repairman in a Dubai skyscraper discovers that the "
            "building's 49th floor, which doesn't officially exist, contains "
            "an exact replica of a small Italian village from 1952. The "
            "residents believe they are still in Italy and have no knowledge "
            "of the outside world. He must decide whether to reveal the "
            "truth or protect their peaceful existence."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original surrealist concept",
    },
]


# -----------------------------------------------------------------------
# METRIC 1: Tool Accuracy
# -----------------------------------------------------------------------

def compute_tool_accuracy(results: list[dict], log: logging.Logger) -> dict:
    """
    Compare the TF-IDF tool's yes/no plagiarism calls against ground truth.

    Plain English:
        - True Positive (TP):  Tool said plagiarism, and it WAS plagiarism.
        - False Positive (FP): Tool said plagiarism, but it WASN'T.
        - True Negative (TN):  Tool said original, and it WAS original.
        - False Negative (FN): Tool said original, but it WAS plagiarism.
    """
    tp = fp = tn = fn = 0

    for r in results:
        pred = r.get("predicted_plagiarism")
        true = r["expected_plagiarism"]
        if pred is None:
            continue
        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and not true:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_scored": total,
    }

    log.info("")
    log.info("METRIC 1: Tool Accuracy (Plagiarism Detection)")
    log.info(f"  Accuracy:    {accuracy:.1%}  ({tp + tn} of {total} correct)")
    log.info(f"  Precision:   {precision:.1%}  (when it says plagiarism, is it right?)")
    log.info(f"  Recall:      {recall:.1%}  (of real plagiarism, how much did it catch?)")
    log.info(f"  Breakdown:   TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return metrics


# -----------------------------------------------------------------------
# METRIC 2: JSON Schema Compliance
# -----------------------------------------------------------------------

def compute_schema_compliance(results: list[dict], log: logging.Logger) -> dict:
    """How many LLM outputs passed Pydantic validation?"""
    valid   = [r for r in results if r["schema_valid"]]
    invalid = [r for r in results if not r["schema_valid"] and r.get("llm_attempted")]
    total_attempted = len(valid) + len(invalid)
    rate = len(valid) / total_attempted if total_attempted else 0

    metrics = {
        "passed": len(valid),
        "failed": len(invalid),
        "total_attempted": total_attempted,
        "compliance_rate": round(rate, 4),
    }

    log.info("")
    log.info("METRIC 2: JSON Schema Compliance")
    log.info(f"  Passed:      {len(valid)} / {total_attempted}  ({rate:.1%})")
    if invalid:
        log.info(f"  Failed IDs:  {[r['id'] for r in invalid]}")

    return metrics


# -----------------------------------------------------------------------
# METRIC 3: Style Adherence (LLM-as-a-Judge)
# -----------------------------------------------------------------------

def judge_style_adherence(
    user_plot: str,
    rewritten_plot: str,
    director: str,
) -> tuple[int, str]:
    """
    Ask an independent LLM call to score how well the rewrite
    captures the target director's filmmaking style.

    Runs locally via Ollama — same model, separate call, acting as
    an independent judge rather than the author.

    Scoring:
        1 = No resemblance to the director's style
        2 = Slight resemblance
        3 = Moderate resemblance
        4 = Strong resemblance
        5 = Unmistakably in the director's style

    Returns:
        (score, justification_string)
    """
    prompt = f"""You are a film studies professor grading student work.

ORIGINAL PLOT:
"{user_plot}"

REWRITTEN PLOT:
"{rewritten_plot}"

TARGET DIRECTOR: {director}

Score the rewrite on how well it captures {director}'s filmmaking style.
Respond with ONLY a JSON object: {{"score": <1-5>, "justification": "<one sentence>"}}

Rubric:
  1 = No resemblance to the director's style
  2 = Slight resemblance
  3 = Moderate resemblance
  4 = Strong resemblance
  5 = Unmistakably in the director's style"""

    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )

    data = json.loads(response.message.content)
    return int(data.get("score", 0)), data.get("justification", "")


def compute_style_scores(results: list[dict], log: logging.Logger) -> dict:
    """Aggregate the LLM-as-a-Judge style scores."""
    scores = [r["style_score"] for r in results if r["style_score"] is not None]

    if not scores:
        log.info("")
        log.info("METRIC 3: Style Adherence - no scores available (LLM calls may have been skipped)")
        return {"mean_score": None, "count": 0}

    mean_s = sum(scores) / len(scores)
    metrics = {
        "mean_score": round(mean_s, 2),
        "min_score": min(scores),
        "max_score": max(scores),
        "count": len(scores),
    }

    log.info("")
    log.info("METRIC 3: Style Adherence (LLM-as-a-Judge, 1 to 5)")
    log.info(f"  Mean score:  {mean_s:.2f} / 5.0")
    log.info(f"  Range:       {min(scores)} - {max(scores)}  (n={len(scores)})")

    return metrics


# -----------------------------------------------------------------------
# Main evaluation runner
# -----------------------------------------------------------------------

def run_evaluation(
    csv_path: str = "movies_dataset.csv",
    run_llm: bool = True,
):
    """
    Run all 50 test cases, compute all metrics, log everything.

    Args:
        csv_path: Path to movie database CSV.
        run_llm:  If False, only runs Step 1 (SBERT+TF-IDF) and skips LLM calls.
                  Useful for testing detection accuracy without Ollama running.
    """
    log = setup_logger("evaluation")

    log.info("=" * 70)
    log.info("CINEMATIC PIPELINE - EVALUATION SUITE")
    log.info(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Test cases:           {len(EVAL_CASES)}")
    log.info(f"Plagiarism threshold: {PLAGIARISM_THRESHOLD}")
    log.info(f"LLM calls enabled:   {run_llm}")
    log.info("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Movie database not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Pre-compute SBERT + TF-IDF features once (not per test case)
    print(f"Initializing NLP features (SBERT + TF-IDF)...")
    import time as _time
    _t0 = _time.time()
    init_nlp(df)
    print(f"Init done in {_time.time() - _t0:.1f}s. Running {len(EVAL_CASES)} test cases...\n")

    results = []

    for i, case in enumerate(EVAL_CASES):
        log.info("")
        log.info(f"--- Case {i + 1}/{len(EVAL_CASES)}: {case['id']} [{case['tier']}] ---")
        log.info(f"  Plot:     \"{case['user_plot'][:80]}...\"")
        log.info(f"  Expected: {'PLAGIARISM' if case['expected_plagiarism'] else 'ORIGINAL'}")

        result = {
            "id": case["id"],
            "tier": case["tier"],
            "expected_plagiarism": case["expected_plagiarism"],
            "predicted_plagiarism": None,
            "similarity_score": None,
            "matched_movie": None,
            "assigned_director": None,
            "schema_valid": False,
            "llm_attempted": False,
            "style_score": None,
            "style_justification": None,
            "error": None,
        }

        try:
            # ---- STEP 1: SBERT+TF-IDF detection (always runs, no API needed) ----
            detection = detect_plagiarism(case["user_plot"], df)
            result["predicted_plagiarism"] = detection["detected_plagiarism"]
            result["similarity_score"] = detection["similarity_score"]
            result["matched_movie"] = detection["matched_movie"]
            result["assigned_director"] = detection["assigned_director"]

            correct = detection["detected_plagiarism"] == case["expected_plagiarism"]
            symbol = "CORRECT" if correct else "WRONG"

            log.info(f"  Match:    \"{detection['matched_movie']}\" (score: {detection['similarity_score']})")
            log.info(f"  Predicted: {'PLAGIARISM' if detection['detected_plagiarism'] else 'ORIGINAL'}  [{symbol}]")

            # ---- STEPS 2-4: LLM rewrite + validation (optional) ----
            if run_llm:
                result["llm_attempted"] = True
                pipeline_result = run_pipeline(
                    case["user_plot"], csv_path, log=log,
                    df=df, detection=detection,
                )
                result["schema_valid"] = True

                # ---- METRIC 3: LLM-as-a-Judge ----
                score, justification = judge_style_adherence(
                    case["user_plot"],
                    pipeline_result.rewritten_plot,
                    pipeline_result.assigned_director,
                )
                result["style_score"] = score
                result["style_justification"] = justification
                log.info(f"  Style:    {score}/5 - {justification}")

        except Exception as e:
            result["error"] = str(e)
            log.error(f"  ERROR: {e}")

        results.append(result)
        if run_llm:
            time.sleep(2)  # rate-limit courtesy

    # ---- SUMMARY ----
    log.info("")
    log.info("=" * 70)
    log.info("EVALUATION RESULTS SUMMARY")
    log.info("=" * 70)

    tool_metrics   = compute_tool_accuracy(results, log)
    schema_metrics = compute_schema_compliance(results, log)
    style_metrics  = compute_style_scores(results, log)

    # Per-case table
    log.info("")
    header = f"{'ID':<10} {'Tier':<22} {'Expect':<10} {'Predict':<10} {'Score':<8} {'Movie':<28} {'JSON':<6} {'Style':<6}"
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        log.info(
            f"{r['id']:<10} "
            f"{r['tier']:<22} "
            f"{'PLAG' if r['expected_plagiarism'] else 'ORIG':<10} "
            f"{'PLAG' if r['predicted_plagiarism'] else 'ORIG' if r['predicted_plagiarism'] is not None else '?':<10} "
            f"{r['similarity_score'] or 0:<8.4f} "
            f"{(r['matched_movie'] or '-')[:26]:<28} "
            f"{'PASS' if r['schema_valid'] else '-':<6} "
            f"{r['style_score'] if r['style_score'] is not None else '-'!s:<6}"
        )

    # Save raw results JSON
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"logs/eval_results_{ts}.json"

    report = {
        "run_timestamp": ts,
        "config": {
            "plagiarism_threshold": PLAGIARISM_THRESHOLD,
            "llm_enabled": run_llm,
            "total_cases": len(EVAL_CASES),
        },
        "metrics": {
            "tool_accuracy": tool_metrics,
            "schema_compliance": schema_metrics,
            "style_adherence": style_metrics,
        },
        "cases": results,
    }

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nFull results saved to: {results_path}")
    log.info(f"Log saved to: {getattr(log, 'log_path', 'unknown')}")

    return report


# -----------------------------------------------------------------------
# Standalone execution
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Cinematic Pipeline")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip LLM calls, only test SBERT+TF-IDF plagiarism detection accuracy",
    )
    args = parser.parse_args()

    print("\nRunning Cinematic Pipeline Evaluation Suite...\n")
    report = run_evaluation(run_llm=not args.local_only)

    print("\n--- QUICK SUMMARY ---")
    m = report["metrics"]
    print(f"Tool Accuracy:     {m['tool_accuracy']['accuracy']:.1%}")
    print(f"Schema Compliance: {m['schema_compliance']['compliance_rate']:.1%}")
    if m["style_adherence"].get("mean_score"):
        print(f"Style Adherence:   {m['style_adherence']['mean_score']:.1f} / 5.0")
    print(f"\nDetailed log: logs/")
