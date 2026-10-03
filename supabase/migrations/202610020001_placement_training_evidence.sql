create table if not exists public.placement_training_evidence (
  id bigint generated always as identity primary key,
  student_key uuid not null,
  cohort text not null,
  evidence_at timestamptz not null,
  outcome_at timestamptz not null,
  verified_skills smallint not null check (verified_skills between 0 and 100),
  academics smallint not null check (academics between 0 and 100),
  projects smallint not null check (projects between 0 and 100),
  aptitude smallint not null check (aptitude between 0 and 100),
  communication smallint not null check (communication between 0 and 100),
  interview smallint not null check (interview between 0 and 100),
  placed smallint not null check (placed in (0, 1)),
  outcome_source text not null check (outcome_source = 'observed'),
  created_at timestamptz not null default now(),
  constraint placement_evidence_precedes_outcome check (evidence_at < outcome_at)
);

create index if not exists placement_training_evidence_cohort_idx
  on public.placement_training_evidence (cohort);
create index if not exists placement_training_evidence_student_idx
  on public.placement_training_evidence (student_key);

alter table public.placement_training_evidence enable row level security;
-- No client policies by default. Read training rows from a trusted server-side export job.
