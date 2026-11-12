"""
Demo family used to seed the live app.

Nipa's AI mode is on, so you can log in as Nick (or anyone else) and talk to
her AI twin. It answers from her journals and past chats, e.g. ask
"Are you free next Saturday?" and it knows about her football final.

Timestamps are given as (days_ago, hour) and turned into real dates when the
data is seeded, so the demo always looks recent.
"""

DEMO_PASSWORD = "password"

# user_id, username, display_name, gender, ai_enabled
DEMO_USERS = [
    ("user1", "nipa", "Nipa", "female", True),
    ("user2", "nick", "Nick", "male", False),
    ("user3", "arif", "Arif", "male", False),
    ("user4", "diana", "Diana", "female", False),
]

# (user_id, other_user_id, how user_id sees other_user_id)
DEMO_RELATIONSHIPS = [
    ("user1", "user2", "brother"),
    ("user2", "user1", "sister"),
    ("user1", "user3", "father"),
    ("user3", "user1", "daughter"),
    ("user1", "user4", "mother"),
    ("user4", "user1", "daughter"),
    ("user2", "user3", "father"),
    ("user3", "user2", "son"),
    ("user2", "user4", "mother"),
    ("user4", "user2", "son"),
    ("user3", "user4", "wife"),
    ("user4", "user3", "husband"),
]

# (user_id, days_ago, hour, text)
DEMO_JOURNALS = [
    ("user1", 12, 21, "Went hiking at Sajek with my uni friends this weekend. Woke up at 5 to see the clouds below the hills. Hiking is honestly the only thing that clears my head. I want to do the Bandarban trail next."),
    ("user1", 9, 20, "Flew kites with Nick on the roof today, like when we were kids. My red kite cut two others! Shakrain is my favourite day of the year."),
    ("user1", 6, 22, "Football practice every evening this week. Our inter-university final is next Saturday at 4 pm and coach made me captain. Nervous but excited. Saying no to everything else until the match is done."),
    ("user1", 4, 23, "Things I love: mango season, rainy evenings, Ammu's khichuri. Things I can't stand: karela, waking up late, and people who are late to practice."),
    ("user1", 2, 22, "Data structures exam on Thursday. Graph algorithms are killing me. Nick promised to help me with Dijkstra on Tuesday night."),
    ("user2", 10, 23, "Shipped the payments feature at work today. Long week, but the team pulled it off."),
    ("user2", 3, 22, "Promised Nipa I'd help her with graph algorithms. Need to remember Dijkstra myself first."),
    ("user3", 8, 21, "Diana and I are planning a trip to Cox's Bazar in December for our 25th anniversary. Haven't told the kids yet."),
    ("user3", 5, 7, "Doctor said to walk 30 minutes every morning. Starting tomorrow, for real this time."),
    ("user4", 7, 22, "Nipa looks tired from all the practice. Making her favourite khichuri on Friday."),
    ("user4", 1, 21, "Bought tickets to watch Nipa's final on Saturday. So proud of her."),
]

# (speaker_id, target_id, days_ago, hour, minute, text)
DEMO_CONVERSATIONS = [
    # Nipa and her father
    ("user3", "user1", 11, 9, 10, "Did you reach Sajek safely? Send a photo of the clouds."),
    ("user1", "user3", 11, 9, 25, "Reached Abbu! The view is unreal, sending photos now."),
    ("user3", "user1", 5, 21, 0, "Nipa, want to come for a morning walk with me tomorrow?"),
    ("user1", "user3", 5, 21, 12, "I have practice at 6 every morning this week, Abbu. After the final, I promise!"),
    # Nipa and her brother
    ("user2", "user1", 9, 16, 30, "Roof at 5? The wind is perfect for kites today."),
    ("user1", "user2", 9, 16, 34, "Coming! Bringing the red one, get ready to lose."),
    ("user1", "user2", 3, 19, 5, "Bhaiya, can you help me with Dijkstra's algorithm? Exam on Thursday."),
    ("user2", "user1", 3, 19, 20, "Sure, Tuesday night after dinner. Bring your notes."),
    ("user2", "user1", 1, 20, 40, "How's practice going, captain?"),
    ("user1", "user2", 1, 20, 52, "Tiring but good. Final is Saturday at 4, you're coming right?"),
    ("user2", "user1", 1, 20, 55, "Wouldn't miss it."),
    # Nipa and her mother
    ("user1", "user4", 6, 13, 0, "Ammu, I'll be late every evening this week, practice till 8."),
    ("user4", "user1", 6, 13, 15, "Okay, I'll keep dinner warm. Don't skip lunch."),
    ("user4", "user1", 2, 18, 0, "Making khichuri on Friday, just for you."),
    ("user1", "user4", 2, 18, 9, "Best mother ever! No karela please."),
    # Parents
    ("user3", "user4", 8, 22, 0, "Should we book Cox's Bazar for December? I found a nice hotel by the beach."),
    ("user4", "user3", 8, 22, 10, "Yes! Let's keep it a surprise from the kids for now."),
    ("user4", "user3", 1, 17, 30, "I bought tickets for Nipa's final on Saturday."),
    ("user3", "user4", 1, 17, 45, "Great, I'll leave the office early."),
    # Nick and his parents
    ("user3", "user2", 4, 10, 0, "Nick, can you fix the WiFi router this weekend? It keeps disconnecting."),
    ("user2", "user3", 4, 12, 30, "Sure Abbu, Saturday morning before Nipa's match."),
    ("user2", "user4", 10, 19, 0, "Ammu, we shipped the big feature today! Treat tonight?"),
    ("user4", "user2", 10, 19, 20, "So proud of you. Biryani it is."),
]
