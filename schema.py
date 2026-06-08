from pydantic import BaseModel, Field

class MovieAnalysis(BaseModel):
    detected_plagiarism: bool = Field(description="True if similarity score is >= 0.45, False otherwise")
    matched_movie: str = Field(description="Title of the closest matching movie from the database")
    similarity_score: float = Field(description="Quantitative similarity score calculated via TF-IDF")
    assigned_director: str = Field(description="Director associated with the matched movie, used as style target")
    rewritten_plot: str = Field(description="The user plot rewritten following the assigned director's style")
    stylistic_notes: str = Field(description="Technical documentation explaining the stylistic choices applied")