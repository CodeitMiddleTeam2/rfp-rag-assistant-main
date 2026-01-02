create table public.documents_chunks_smk_4 (
  id uuid not null default gen_random_uuid (),
  chunk_id uuid not null,
  pages integer[] not null,
  announcement_id text null,
  announcement_round integer null,
  project_name text null,
  project_budget bigint null,
  ordering_agency text null,
  published_at timestamp with time zone null,
  bid_start_at timestamp with time zone null,
  bid_end_at timestamp with time zone null,
  text text not null,
  length integer not null,
  content_type text not null,
  chunk_index integer not null,
  source_file text not null,
  file_type text not null,
  metadata jsonb null,
  embedding public.vector null,
  created_at timestamp with time zone not null default now(),
  ngram_text text null,
  constraint documents_chunks_smk_4_pkey1 primary key (id),
  constraint documents_chunks_smk_4_chunk_id_key unique (chunk_id)
) TABLESPACE pg_default;

create unique INDEX IF not exists documents_chunks_smk_4_pkey on public.documents_chunks_smk_4 using btree (id) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_embedding_idx on public.documents_chunks_smk_4 using ivfflat (embedding vector_cosine_ops)
with
  (lists = '100') TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_metadata_gin_idx on public.documents_chunks_smk_4 using gin (metadata) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_pages_gin_idx on public.documents_chunks_smk_4 using gin (pages) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_content_type_idx on public.documents_chunks_smk_4 using btree (content_type) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_ordering_agency_idx on public.documents_chunks_smk_4 using btree (ordering_agency) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_project_name_idx on public.documents_chunks_smk_4 using btree (project_name) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_announcement_id_idx on public.documents_chunks_smk_4 using btree (announcement_id) TABLESPACE pg_default;

----------------------------------------------------------------------------------------------
create table public.documents_chunks_smk_4 (
  id uuid not null default gen_random_uuid (),
  chunk_id uuid not null,
  pages integer[] not null,
  announcement_id text null,
  announcement_round integer null,
  project_name text null,
  project_budget bigint null,
  ordering_agency text null,
  published_at timestamp with time zone null,
  bid_start_at timestamp with time zone null,
  bid_end_at timestamp with time zone null,
  text text not null,
  length integer not null,
  content_type text not null,
  chunk_index integer not null,
  source_file text not null,
  file_type text not null,
  metadata jsonb null,
  embedding public.vector null,
  created_at timestamp with time zone not null default now(),
  ngram_text text null,
  constraint documents_chunks_smk_4_pkey1 primary key (id),
  constraint documents_chunks_smk_4_chunk_id_key unique (chunk_id)
) TABLESPACE pg_default;

create unique INDEX IF not exists documents_chunks_smk_4_pkey on public.documents_chunks_smk_4 using btree (id) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_embedding_idx on public.documents_chunks_smk_4 using ivfflat (embedding vector_cosine_ops)
with
  (lists = '100') TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_metadata_gin_idx on public.documents_chunks_smk_4 using gin (metadata) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_pages_gin_idx on public.documents_chunks_smk_4 using gin (pages) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_content_type_idx on public.documents_chunks_smk_4 using btree (content_type) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_ordering_agency_idx on public.documents_chunks_smk_4 using btree (ordering_agency) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_project_name_idx on public.documents_chunks_smk_4 using btree (project_name) TABLESPACE pg_default;

create index IF not exists documents_chunks_smk_4_announcement_id_idx on public.documents_chunks_smk_4 using btree (announcement_id) TABLESPACE pg_default;