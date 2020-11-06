open Base

let load_data lines =
  let line_split line =
    line
    |> String.lsplit2_exn ~on:' '
    |> (fun (num, review) ->
        Int.of_string num, review) in
  lines
  |> List.map ~f:line_split
  |> List.unzip

let () =
  Stdlib.print_endline "dl"
