library(tidyverse)
library(syuzhet)
library(tidytext)


dhaka_reviews_df <- read.csv("./data_science2/restaurant_review_in dhaka.csv")
head(dhaka_reviews_df)
dhaka_df <- tibble(text=tolower(dhaka_reviews_df$Review.Text))
dhaka <- data.frame(line=1:nrow(dhaka_df), text=dhaka_df$text)
head(dhaka_df)
head(dhaka)

#Using syuzhet package based on the NRC sentiment dictionary
reviews <- dhaka[1:700,c(1,2)]
dim(reviews)

emotions <- get_nrc_sentiment(reviews$text)
emo_bar <- colSums(emotions)
emo_sum <- data.frame(count=emo_bar, emotions=names(emo_bar))
sort_emotions <- emo_sum[order(emo_sum$count, decreasing=TRUE), ]

#Visualizing
ggplot(sort_emotions, aes(x=reorder(emotions, -count), y=count, fill=emotions)) +
  geom_bar(stat='identity')
  


#Using tidytext package using the bing lexicon
bing_word_counts <- dhaka %>%
  unnest_tokens(word, text)%>%
  inner_join(get_sentiments("bing"))%>%
  count(word, sentiment, sort=T)

top_10_bing_word_counts <- bing_word_counts%>%
  group_by(sentiment)%>%
  slice_max(order_by = n, n=10)%>%
  ungroup()%>%
  mutate(word=reorder(word, n))
top_10_bing_word_counts  

#create a barplot
top_10_bing_word_counts  %>%
  ggplot(aes(word, n, fill=sentiment)) +
  geom_col(show.legend=F)+
  facet_wrap(~sentiment, scales="free_y")+
  labs(y="Contribution to sentiment", x=NULL)+
  coord_flip()
  

#tidytext package using the loughran lexicon
loughran_word_counts <- dhaka%>%
  unnest_tokens(word, text) %>%
  inner_join(get_sentiments("loughran"))%>%
  count(word, sentiment, sort=T)
loughran_word_counts

#Select top 10 words by sentiment
top_10_loughran_word_counts <- loughran_word_counts%>%
  group_by(sentiment)%>%
  slice_max(order_by=n, n=10) %>%
  ungroup()%>%
  mutate(word=reorder(word, n))
top_10_loughran_word_counts

#Create a barplot 
top_10_loughran_word_counts  %>%
  ggplot(aes(word, n, fill=sentiment)) +
  geom_col(show.legend=F)+
  facet_wrap(~sentiment, scales="free_y")+
  labs(y="Contribution to sentiment", x=NULL)+
  coord_flip()
