CREATE OR REPLACE FUNCTION public.match_documents_chunks_structural_pjw_vector(query_embedding vector, match_threshold double precision, match_count integer)
 RETURNS TABLE(chunk_id uuid, announcement_id text, announcement_round integer, project_name text, project_budget bigint, ordering_agency text, published_at text, bid_start_at text, bid_end_at text, text text, source_file text, file_type text, length integer, metadata jsonb, score double precision)
 LANGUAGE sql
 STABLE
AS $function$
  select
    d.chunk_id,
    d.announcement_id,
    d.announcement_round,
    d.project_name,
    d.project_budget,
    d.ordering_agency,
    d.published_at,
    d.bid_start_at,
    d.bid_end_at,
    d.text,
    d.source_file,
    d.file_type,
    d.length,
    d.metadata,
    (1 - (d.embedding <=> query_embedding)) as score
  from public.documents_chunks_structural_pjw d
  where d.embedding is not null
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by d.embedding <=> query_embedding
  limit match_count;
$function$

CREATE OR REPLACE FUNCTION public.match_documents_chunks_structural_pjw_bm25(query text, match_count integer)
 RETURNS TABLE(chunk_id uuid, announcement_id text, announcement_round integer, project_name text, project_budget bigint, ordering_agency text, published_at text, bid_start_at text, bid_end_at text, text text, source_file text, file_type text, length integer, metadata jsonb, score double precision)
 LANGUAGE sql
 STABLE
AS $function$
  select
    d.chunk_id,
    d.announcement_id,
    d.announcement_round,
    d.project_name,
    d.project_budget,
    d.ordering_agency,
    d.published_at,
    d.bid_start_at,
    d.bid_end_at,
    d.text,
    d.source_file,
    d.file_type,
    d.length,
    d.metadata,
    ts_rank_cd(
      to_tsvector('simple', coalesce(d.text,'')),
      websearch_to_tsquery('simple', query)
    ) as score
  from public.documents_chunks_structural_pjw d
  where to_tsvector('simple', coalesce(d.text,'')) @@ websearch_to_tsquery('simple', query)
  order by score desc
  limit match_count;
$function$

----------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.match_documents_chunks_smk4_vector(query_embedding vector, match_threshold double precision, match_count integer)
 RETURNS TABLE(chunk_id uuid, announcement_id text, announcement_round integer, project_name text, project_budget bigint, ordering_agency text, published_at timestamp with time zone, bid_start_at timestamp with time zone, bid_end_at timestamp with time zone, text text, length integer, source_file text, file_type text, metadata jsonb, score double precision)
 LANGUAGE sql
 STABLE
AS $function$
  select
    d.chunk_id,
    d.announcement_id,
    d.announcement_round,
    d.project_name,
    d.project_budget,
    d.ordering_agency,
    d.published_at,
    d.bid_start_at,
    d.bid_end_at,
    d.text,
    d.length,
    d.source_file,
    d.file_type,
    d.metadata,
    (1 - (d.embedding <=> query_embedding))::double precision as score
  from public.documents_chunks_smk_4 d
  where d.embedding is not null
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by (d.embedding <=> query_embedding) asc
  limit match_count;
$function$

CREATE OR REPLACE FUNCTION public.match_documents_chunks_smk4_bm25_ngram(query text, match_count integer)
 RETURNS TABLE(chunk_id uuid, announcement_id text, announcement_round integer, project_name text, project_budget bigint, ordering_agency text, published_at text, bid_start_at text, bid_end_at text, text text, source_file text, file_type text, length integer, metadata jsonb, score double precision)
 LANGUAGE sql
 STABLE
AS $function$
WITH q AS (
  SELECT
    regexp_replace(
      public.make_ngram(query, 2),
      '\s+',
      ' | ',
      'g'
    ) AS tsquery
)
SELECT
  d.chunk_id,
  d.announcement_id,
  d.announcement_round,
  d.project_name,
  d.project_budget,
  d.ordering_agency,
  d.published_at,
  d.bid_start_at,
  d.bid_end_at,
  d.text,
  d.source_file,
  d.file_type,
  d.length,
  d.metadata,
  ts_rank_cd(
    to_tsvector('simple', d.ngram_text),
    to_tsquery('simple', q.tsquery)
  ) AS score
FROM public.documents_chunks_smk_4 d
CROSS JOIN q
WHERE to_tsvector('simple', d.ngram_text)
      @@ to_tsquery('simple', q.tsquery)
ORDER BY score DESC
LIMIT match_count;
$function$

