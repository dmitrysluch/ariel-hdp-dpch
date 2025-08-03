TBD

In short you have to make AsyncClient mixin and AsyncClient implementation. Than you can make a DataFrame by overloading only methods which actually require query to the server. BaseDataFrame is common for DataFrame and AsyncDataFrame