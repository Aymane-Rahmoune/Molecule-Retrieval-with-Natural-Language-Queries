# Molecule-Retrieval-with-Natural-Language-Queries

This project describes the approach and the methods we used for the challenge of retrieving
molecules using natural language queries, as presented in the ALTEGRAD
2023 Data Challenge. We investigate the integration of two very different data
modalities—natural language and molecular structures (graphs)—to address the
complex task of identifying relevant molecules based on textual descriptions. Our
approach leverages contrastive self-supervised learning to co-train a text encoder
and a molecule encoder, aiming to bridge the semantic gap between text descriptions
and molecular graph representations. By mapping text-molecule pairs into a
shared representation space, we facilitate the retrieval of molecules corresponding
to natural language queries without direct reference information.