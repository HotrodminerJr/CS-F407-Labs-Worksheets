% ---------------------------------------------------------------
% Optional extension: Prolog as an independent plan verifier
% (Tasks 6, 7, 8 of the lab)
%
% Run with SWI-Prolog:
%     swipl planner.pl
%     ?- can_move(a, b).            % Task 6
%     ?- can_move(a, c).            % Task 6 — should be `false`
%     ?- valid_move(a, b).          % Task 7
%     ?- valid_move(b, c).          % Task 7
%     ?- valid_move(a, c).          % Task 7 — should be `false`
%     ?- reduce_speed.              % Task 8
%     ?- verify_plan([move(a,b), move(b,c)]).  % check a full plan
% ---------------------------------------------------------------

% ---- Warehouse topology (facts) ----
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% ---- Task 6: can the robot move between two locations? ----
% Rule: a single-step move is possible iff the two locations are connected.
can_move(X, Y) :- connected(X, Y).

% ---- Task 7: validate one proposed move against the warehouse ----
valid_move(X, Y) :- connected(X, Y).

% Verify a whole plan (list of move/2 terms). Succeeds iff every step is
% supported by the warehouse facts. This is the "independent verification"
% used to double-check whatever plan the Python planner produced.
verify_plan([]).
verify_plan([move(X, Y) | Rest]) :-
    valid_move(X, Y),
    verify_plan(Rest).

% ---- Task 8: Facts + Rules -> Inference -> Query Answer ----
% The chain is:
%     wet_road           (a fact)
%     wet_road -> slippery
%     slippery -> reduce_speed
% so Prolog can conclude `reduce_speed` from the single fact `wet_road`.
wet_road.
slippery      :- wet_road.
reduce_speed  :- slippery.
