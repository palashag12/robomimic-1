"""
Set of global variables shared across robomimic
"""
# Sets debugging mode. Should be set at top-level script so that internal
# debugging functionalities are made active
DEBUG = False


### Slack Notifications ###

# Token for sending slack notifications
SLACK_TOKEN = None

# User ID for user that should receive slack notifications
SLACK_USER_ID = None