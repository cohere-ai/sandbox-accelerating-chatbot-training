
```
################################################################################
#    ____      _                     ____                  _ _                 #
#   / ___|___ | |__   ___ _ __ ___  / ___|  __ _ _ __   __| | |__   _____  __  #
#  | |   / _ \| '_ \ / _ \ '__/ _ \ \___ \ / _` | '_ \ / _` | '_ \ / _ \ \/ /  #
#  | |__| (_) | | | |  __/ | |  __/  ___) | (_| | | | | (_| | |_) | (_) >  <   #
#   \____\___/|_| |_|\___|_|  \___| |____/ \__,_|_| |_|\__,_|_.__/ \___/_/\_\  #
#                                                                              #
# This project is part of Cohere Sandbox, Cohere's Experimental Open Source    #
# offering. This project provides a library, tooling, or demo making use of    #
# the Cohere Platform. You should expect (self-)documented, high quality code  #
# but be warned that this is EXPERIMENTAL. Therefore, also expect rough edges, #
# non-backwards compatible changes, or potential changes in functionality as   #
# the library, tool, or demo evolves. Please consider referencing a specific   #
# git commit or version if depending upon the project in any mission-critical  #
# code as part of your own projects.                                           #
#                                                                              #
# Please don't hesitate to raise issues or submit pull requests, and thanks    #
# for checking out this project!                                               #
#                                                                              #
################################################################################
```

**Maintainer:** [stewart-co](https://github.com/stewart-co) \
**Project maintained until at least (YYYY-MM-DD):** 2023-04-01

# Route Generation
Modern chatbots typically need a combination of intents/routes and associated training data to perform tasks well enough to be deployed. Route generation takes that task and automates it by leveraging the general knowledge of large language models. By taking a natural language description of an intent (e.g. this intent is for users who need to reset their password), the language model can go and create example phrases that match that description. This takes the work out of sourcing large amounts of data simply to test a new intent or route.

These example phrases can be used with embeddings to map incoming messages from users based on similarity, or used in downstream chatbots as additional training data.

# How it works
Once a user passes in a few route descriptions, the [Cohere generate endpoint](https://docs.cohere.ai/generate-reference/) is used to create some example messages relating to that description. Once all of the examples have been created, [Cohere embed](https://docs.cohere.ai/embed-reference) is used to get representations of those examples. By clustering those representations and comparing incoming messages to its nearest neighbors, new messages can be assigned to a route.

![how-it-works](https://user-images.githubusercontent.com/108292383/200897059-db3f44a2-861e-4d8b-bd33-e7919b670887.png)

# Installation
This repository uses [Cohere](https://docs.cohere.ai/)’s large language models that allow complex generation, classification and representation of text; full documentation available at [docs.cohere.ai](https://docs.cohere.ai/).

To deploy this app <yourself>, you will need an API key.  Sign up to get a free non-production API key at [cohere.ai](https://dashboard.cohere.ai/welcome/register?utm_source=github&utm_medium=content&utm_campaign=sandbox&utm_content=routegeneration)

1- Clone the repository.

2- Install all the dependencies:

```pip install -r requirements.txt```

3- Create a `secrets.toml` file to store api keys and put it in the `.streamlit` folder
```
cohere_api_token = "{API KEY}"
```

4- Running the streamlit demo
Try the demo by running the Streamlit app

```streamlit run route-generation/front_end.py```

# Get support
If you have any questions or comments, please file an issue or reach out to us on [Discord](https://discord.gg/co-mmunity).

# Contributors
If you would like to contribute to this project, please read `CONTRIBUTORS.md`
in this repository, and sign the Contributor License Agreement before submitting
any pull requests. A link to sign the Cohere CLA will be generated the first time 
you make a pull request to a Cohere repository.

# License
route-generation has an MIT license, as found in the LICENSE file.
