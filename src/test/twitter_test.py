import twitter

# XXX: Go to http://dev.twitter.com/apps/new to create an app and get values
# for these credentials, which you'll need to provide in place of these
# empty string values that are defined as placeholders.
# See https://dev.twitter.com/docs/auth/oauth for more information 
# on Twitter's OAuth implementation.

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
OAUTH_TOKEN =''
OAUTH_TOKEN_SECRET = ''

auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)

twitter_api = twitter.Twitter(auth=auth)

# Nothing to see by displaying twitter_api except that it's now a
# defined variable

print twitter_api

# The Yahoo! Where On Earth ID for the entire world is 1.
# See https://dev.twitter.com/docs/api/1.1/get/trends/place and
# http://developer.yahoo.com/geo/geoplanet/

WORLD_WOE_ID = 1
US_WOE_ID = 23424977
SOUTH_KOREA = 23424868 # south korea

# Prefix ID with the underscore for query string parameterization.
# Without the underscore, the twitter package appends the ID value
# to the URL itself as a special case keyword argument.
world_trends = twitter_api.trends.place(_id=WORLD_WOE_ID)
korea_trends = twitter_api.trends.place(_id=SOUTH_KOREA)
us_trends = twitter_api.trends.place(_id=US_WOE_ID)

# print world_trends
# print
# print korea_trends
# print
# print us_trends

import json

# print json.dumps(world_trends, indent=1)
# print json.dumps(korea_trends, indent=1)
# print json.dumps(us_trends, indent=1)

korea_trends_set = set([trend['name'] for trend in korea_trends[0]['trends']])
us_trends_set = set([trend['name'] for trend in us_trends[0]['trends']]) 
common_trends = korea_trends_set.intersection(us_trends_set)

# print common_trends
trendslist = list(korea_trends_set)
for cnt in range(len(trendslist)):
    print trendslist[cnt]


# q = '#MentionSomeoneImportantForYou' 
import json
q = '기아차 실적' 
count = 100


# See https://dev.twitter.com/docs/api/1.1/get/search/tweets
search_results = twitter_api.search.tweets(q=q, since_id="488516059096313800",lang = "ko",count=count)
statuses = search_results['statuses']
# print "Length of statuses 1", len(statuses)
print json.dumps(statuses,indent = 1)

for _ in range(5):
    print "Length of statuses 2", len(statuses)
    try:
        next_results = search_results['search_metadata']['next_results']
    except KeyError, e: # No more results when next_results doesn't exist
        break
        
    # Create a dictionary from next_results, which has the following form:
    # ?max_id=313519052523986943&q=NCAA&include_entities=1
    kwargs = dict([ kv.split('=') for kv in next_results[1:].split("&") ])
    
    search_results = twitter_api.search.tweets(**kwargs)
    statuses += search_results['statuses']


status_texts = [ status['text'] for status in statuses ]
status_createat = [ status['created_at'] for status in statuses ]

tweets_texts = list(set(status_texts))
for cnt in range(len(tweets_texts)):
    print tweets_texts[cnt]

tweets_createat = list(set(status_createat))
for cnt in range(len(tweets_createat)):
    print tweets_createat[cnt]
