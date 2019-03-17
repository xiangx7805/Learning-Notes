# LDA model :thought_balloon:
LDA represents documents as mixtures of topics that spit out words with certain probabilities. It assumes that documents are produced in the following fashion: when writing each document, you

*  Decide on the number of words N the document will have (say, according to a Poisson distribution).  
*  Choose a topic mixture for the document (according to a Dirichlet distribution over a fixed set of K topics). For example, assuming that we have the two food and cute animal topics, you might choose the document to consist of 1/3 food and 2/3 cute animals.  
*  Generate each word *W* in the document by:  
   *  First picking a topic (according to the multinomial distribution that you sampled above; for example, you might pick the food topic with 1/3 probability and the cute animals topic with 2/3 probability).  
   *  Using the topic to generate the word itself (according to the topic’s multinomial distribution). For example, if we selected the food topic, we might generate the word “broccoli” with 30% probability, “bananas” with 15% probability, and so on.

Assuming this generative model for a collection of documents, LDA then tries to backtrack from the documents to find a set of topics that are likely to have generated the collection.

# LDA & Topic Modeling :sunny:

## How does it work? :paw_prints:

Say we’ve got a collection of documents, and we want to identify underlying “topics” that organize the collection. Assume that each document contains a mixture of different topics. Let’s also assume that a “topic” can be understood as a collection of words that have different probabilities of appearance in passages discussing the topic.

Of course, we can’t directly observe topics; in reality all we have are documents. Topic modeling is a way of extrapolating backward from a collection of documents to infer the discourses (“topics”) that could have generated them.

Unfortunately, there is no way to infer the topics exactly: there are too many unknowns. But pretend for a moment that we had the problem mostly solved. Suppose we knew which topic produced every word in the collection, except for this one word in document *D*.

:question: How are we going to decide whether this occurrence of W belongs to topic *Z*?

We can’t know for sure. But one way to guess is to consider two questions.
  A) How often does word *W* appear in topic *Z* elsewhere? If *W* often occurs in discussions of *Z*, then this instance of *W* might belong to *Z* as well. :grey_exclamation: But a word can be common in more than one topic. :point_right: So next we consider
  B) How common is topic *Z* in the rest of this document?

:cactus: Here’s what we’ll do.   
For each possible topic *Z*, we’ll multiply the frequency of this word type *W* in *Z* by the number of other words in document *D* that already belong to *Z*. The result will represent the probability that this word came from *Z*. Here’s the actual formula:    
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cfn_cs%20P%28%20Z%20%5Cmid%20W%2CD%29%3D%20%5Cfrac%7B%5C%23%20%7E%20of%20%7E%20word%20%7E%20W%20%7E%20in%20%7E%20topic%20Z%20%7E&plus;%20%7E%20%5Cbeta_%7Bw%7D%7D%7B%7B%20total%20%7E%20tokens%20%7E%20in%20%7E%20Z%20%7E%20&plus;%20%7D%7E%20%5Cbeta%7D%20*%20%28%5Ctext%7B%5C%23%20of%20words%20in%20D%20that%20belongs%20to%20Z%20&plus;%20%7D%5Calpha%20%29)

> Greek letters -- *“hyperparameters”* OR *fudge factors*.   
**There’s some chance that this word belongs to topic Z even if it is nowhere else associated with Z; the fudge factors keep that possibility open.** 

The overall emphasis on probability in this technique, so it’s called **probabilistic topic modeling**.

:exclamation: By improvement, our model will gradually become more consistent as topics focus on specific words and documents. **But** can’t ever become perfectly consistent, because words and documents don’t line up in one-to-one fashion. So the tendency for topics to concentrate on particular words and documents will eventually be limited by the actual, messy distribution of words across documents.

That’s how topic modeling works in practice. You assign words to topics randomly and then just keep improving the model, to make your guess more internally consistent, until the model reaches an equilibrium that is as consistent as the collection allows.

## What is it for? :notes:

Topic modeling gives us a way to infer the latent structure behind a collection of documents. This technique becomes more useful as we move toward a scale that is too large to fit into human memory.

> :cop: **A bit of tuning required up front**  
In particular, a standard list of `stopwords` is rarely adequate.   
For instance, in topic-modeling fiction it's useful to get rid of *the most common personal pronouns* and *personal names*.  
This sort of thing is very much a critical judgment call; it’s not a science.


> :man: **author signal**
Topics that can be dominated by a single author, and clearly reflect her unique idiom.   
This could be a feature or a bug, depending on your interests.  
But the author signal may diffuse more or less automatically as the collection expands.


## What are the limits of probabilistic topic modeling?:skull:
LDA seemed like a fragile and unnecessarily complicated technique.
*  Not sure how much value they will have as evidence.  
*  Require you to make a series of judgment calls that deeply shape the results you get (from choosing stopwords, to the number of topics produced, to the scope of the collection).  
*  The resulting model ends up being tailored in difficult-to-explain ways by a researcher’s preferences. Simpler techniques, like **corpus comparison**, can answer a question more transparently and persuasively, if the question is already well-formed.  
*  Moreover, probabilistic techniques have an unholy thirst for memory and processing time.   
*  Probabilistic methods are also less robust than, say, vector-space methods. **LDA is sensitive to noise**, after all, because it is sensitive to everything else!

:feet: On the whole, if you’re just fishing for interesting patterns in a large collection of documents, probabilistic techniques are the way to go.

## Where to go next

* Examples
  *  The standard implementation of LDA is the one in MALLET.
  *  [Tedunderwood's example](https://github.com/tedunderwood/BrowseLDA) models on collections of 18/19c volumes.

* If want to understand the technique more deeply and tinker with the algorithm, read up on [Bayesian statistics](http://en.wikipedia.org/wiki/Bayesian_probability).

* David Blei invented LDA, and writes well -- To understand why this technique has “Dirichlet” in its name, his works are the next things to read. e.g. [*Introduction to Probabilistic Topic Models*](http://www.cs.princeton.edu/~blei/publications.html).

* [*Rethinking LDA: Why Priors Matter*](http://people.cs.umass.edu/~mimno/publications.html), a really thoughtful article by Hanna Wallach, David Mimno, and Andrew McCallum that explains the “hyperparameters” .

* A whole family of techniques related to LDA :
  * Topics Over Time
  *  Dynamic Topic Modeling
  *  Hierarchical LDA
  *  Pachinko Allocation — that one can explore rapidly enough by searching the web.   
  In general, it’s a good idea to approach these skeptically. They all promise to do more than LDA does, but they also add additional assumptions to the model.

  

# Article Refrence 
1. [Introduction to Latent Dirichlet Allocation](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/)
2. [Topic modeling made just simple enough.](https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/)
