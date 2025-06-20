import copy
from citation_utils import find_quote, highlight_source

class Answer(object):
    def __init__(self, 
                 question: str,
                 cited_response_numbered_for_all_quotes: str,
                 all_quotes: list[str],
                 all_sources: list[str],
                 requirement_satisfied: bool,
                 answer_dict: dict = {}):
        self.question = question
        self.cited_response_numbered_for_all_quotes = cited_response_numbered_for_all_quotes
        self.all_quotes = all_quotes
        self.all_sources = all_sources
        self.requirement_satisfied = requirement_satisfied
        self.answer_dict = answer_dict

        # Maps citation numbers from the case with all quotes to the case with cited quotes
        # Identify the cited quotes from the response
        self.citation_number_mapping, self.cited_quotes = self._identify_cited_quotes()
        
        # Response with citations re-numbered to be consistent with self.cited_quotes
        self.cited_response = self._renumber_response_citations()

        # List of cited sources with the cited quotes with <highlight> tags
        self.tag_highlighted_cited_sources = self._tag_highlight_sources(self.all_sources, self.cited_quotes)
        
        # List of cited sources with the cited quotes with color highlights
        self.color_highlighted_cited_sources = self._color_highlight_cited_sources()

        # TODO List of names of the sources that were cited in the response
        # self.cited_source_names = 

    def _identify_cited_quotes(self): 
        """Identify the quotes that were cited in the response in cited_quotes
           and create a mapping of citation numbers to indices in citation_number_mapping.""" 
        cited_quote_indices = []
        citation_number_mapping = {} 
        for i in range(len(self.all_quotes)):
            if f"[{i+1}]" in self.cited_response_numbered_for_all_quotes:
                cited_quote_indices.append(i)
                citation_number_mapping[i+1] = len(cited_quote_indices)
        cited_quotes = [self.all_quotes[i] for i in cited_quote_indices]
        return citation_number_mapping, cited_quotes
    
    def _renumber_response_citations(self):
        """Renumber the citations in the response to match the cited quotes."""
        cited_response = self.cited_response_numbered_for_all_quotes
        for key in  self.citation_number_mapping.keys():
            cited_response = cited_response.replace(f"[{key}]", f"[{self.citation_number_mapping[key]}]")
        return cited_response
    
    def _tag_highlight_sources(self, sources, quotes):
        """Highlight the quotes in the sources with <highlight> tags."""
        tag_highlighted_cited_sources = copy.deepcopy(sources)  
        for quote in quotes:
            for i, source in enumerate(tag_highlighted_cited_sources):
                if find_quote(quote, source):
                    highlighted_source = highlight_source(quote, source, f"<highlight>{quote}</highlight>")
                    if (highlighted_source):
                        tag_highlighted_cited_sources[i] = highlighted_source
                    else:
                        # Note: improve highlighting if this case proves common. This case will be hit if the quotes are overlapping
                        print(f"!!! Warning: Quote '{quote}' could not be highlighted in source {i}. It was still verified to be in the source and was used in generation.")
                    break
        # remove any sources that did not have a quote highlighted
        tag_highlighted_cited_sources = [source for source in tag_highlighted_cited_sources if "<highlight>" in source]
        return tag_highlighted_cited_sources

    def _color_highlight_cited_sources(self):
        """Convert the tag highlighted sources to color-coded strings for terminal output."""
        color_highlighted_sources = []
        for source in self.tag_highlighted_cited_sources:
            source = source.replace("<highlight>", "\033[92m")
            source = source.replace("</highlight>", "\033[0m")
            color_highlighted_sources.append(source)
        return color_highlighted_sources
            
    def __str__(self):
        return f"Answer(question={self.question}, cited_response={self.cited_response}, all_source_quotes={self.all_source_quotes})"

    def __repr__(self):
        return self.__str__()