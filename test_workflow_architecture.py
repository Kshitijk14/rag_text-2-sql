from llama_index.core.workflow import Workflow, step, StartEvent, Event
from llama_index.core.workflow.events import StopEvent
# from dataclasses import dataclass
import asyncio

# @dataclass
class TablesEvent(Event):
    query: str

# @dataclass
class SQLQueryEvent(Event):
    sql: str

# @dataclass
class FinalAnswer(Event):
    answer_str: str


class MiniCtxWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx = {}  # shared dict

    @step
    def retrieve_tables(self, ev: StartEvent) -> TablesEvent:
        self.ctx["retrieved_table_count"] = 3
        self.ctx["retrieved_preview"] = "users, orders, payments"
        return TablesEvent(query=ev.input)

    @step
    def text2sql(self, ev: TablesEvent) -> SQLQueryEvent:
        sql = f"SELECT * FROM users WHERE name LIKE '%{ev.query}%'"
        self.ctx["final_sql_query"] = sql
        return SQLQueryEvent(sql=sql)

    @step
    def synthesize_answer(self, ev: SQLQueryEvent) -> FinalAnswer:
        self.ctx["row_count"] = 2
        answer = f"Found 2 users matching your search for '{ev.sql}'."
        return FinalAnswer(answer_str=answer)

    @step
    def stop(self, ev: FinalAnswer) -> StopEvent:
        return StopEvent(result={
            "answer": ev.answer_str,
            "debug": dict(self.ctx)
        })

async def test_run():
    wf = MiniCtxWorkflow(timeout=30)
    result = await wf.run(input="Alice")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_run())


# OUTPUT
"""
(rag) C:\Users\Hp\Documents\GitHub\rag_text-2-sql>python test_workflow.py
{
    'answer': "Found 2 users matching your search for 'SELECT * FROM users WHERE name LIKE '%Alice%''.", 
    'debug': {
        'retrieved_table_count': 3, 
        'retrieved_preview': 'users, orders, payments', 
        'final_sql_query': "SELECT * FROM users WHERE name LIKE '%Alice%'", 
        'row_count': 2
    }
}
"""